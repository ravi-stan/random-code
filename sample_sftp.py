#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export selected parts of traces from an Arize Phoenix project.

Features
--------
- Robust CLI with clear errors and logging
- Works with Phoenix Cloud (API key) or self-hosted (base URL)
- Filters traces by *root span name*
- Exports selected attributes (e.g., input.value, output.value)
- Optional "child join" pattern to export data from child spans
  (e.g., retrieved documents) joined back to the root span
- Writes Parquet or CSV

Usage
-----
# Root-only export: inputs and outputs for traces whose root name matches
python phoenix_export.py \
  --project my-llm-app \
  --trace-name "rag_query" \
  --select input=input.value --select output=output.value \
  --out traces.parquet

# Export retrieved documents from retriever child spans for those traces
python phoenix_export.py \
  --project my-llm-app \
  --trace-name "rag_query" \
  --child-kind RETRIEVER \
  --explode "retrieval.documents:reference=document.content" \
  --select input=input.value \
  --out retrieved_docs.csv

Environment
-----------
Phoenix client will auto-read base URL & headers if set:
  PHOENIX_COLLECTOR_ENDPOINT (or base_url via --base-url)
  PHOENIX_CLIENT_HEADERS (e.g., "api_key=..."), or pass --api-key

Docs this script follows:
- Client + spans dataframe APIs: https://arize-phoenix.readthedocs.io/ (spans.get_spans_dataframe)
- SpanQuery DSL (.where/.select/.explode, joining via parent_id): Arize docs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Phoenix imports (try modern client types first, then fallback)
try:
    from phoenix.client import Client  # modern client
    try:
        # newer packaging
        from phoenix.client.types.spans import SpanQuery  # type: ignore
    except Exception:  # pragma: no cover
        # older packaging
        from phoenix.trace.dsl import SpanQuery  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: arize-phoenix. Install it: `pip install arize-phoenix`"
    ) from e


@dataclass(frozen=True)
class SelectSpec:
    alias: str
    expr: str  # e.g., "input.value" or "output.value"


@dataclass(frozen=True)
class ExplodeSpec:
    list_attr: str                      # e.g., "retrieval.documents"
    alias_to_expr: Dict[str, str]       # e.g., {"reference": "document.content", "score": "document.score"}


def parse_select(values: List[str]) -> List[SelectSpec]:
    """
    Parse --select alias=span_attr entries.
    Example: --select input=input.value --select output=output.value
    """
    out: List[SelectSpec] = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"--select must look like alias=expr, got: {v}")
        alias, expr = v.split("=", 1)
        alias, expr = alias.strip(), expr.strip()
        if not alias or not expr:
            raise ValueError(f"Invalid --select entry: {v}")
        out.append(SelectSpec(alias=alias, expr=expr))
    return out


def parse_explode(values: List[str]) -> List[ExplodeSpec]:
    """
    Parse --explode of the form:
      <list_attr>:<alias>=<expr>[,<alias>=<expr>...]
    Example:
      --explode "retrieval.documents:reference=document.content,score=document.score"
    """
    specs: List[ExplodeSpec] = []
    for v in values:
        if ":" not in v:
            raise ValueError(
                f"--explode must be '<list_attr>:alias=expr[,alias=expr...]', got: {v}"
            )
        left, right = v.split(":", 1)
        list_attr = left.strip()
        pairs = [p.strip() for p in right.split(",") if p.strip()]
        alias_to_expr: Dict[str, str] = {}
        for p in pairs:
            if "=" not in p:
                raise ValueError(f"Bad alias=expr in --explode: {p}")
            alias, expr = p.split("=", 1)
            alias_to_expr[alias.strip()] = expr.strip()
        if not list_attr or not alias_to_expr:
            raise ValueError(f"Invalid --explode entry: {v}")
        specs.append(ExplodeSpec(list_attr=list_attr, alias_to_expr=alias_to_expr))
    return specs


def build_root_query(trace_name: str,
                     selects: List[SelectSpec],
                     explodes: List[ExplodeSpec]) -> SpanQuery:
    """
    Build a SpanQuery that:
      - filters to root spans whose name == trace_name
      - selects requested attributes
      - optional explode(s)

    Root spans are also enforced by root_spans_only=True at call time.
    """
    # safer literal for name equality
    name_literal = json.dumps(trace_name)

    q = SpanQuery().where(f"name == {name_literal}")
    if selects:
        q = q.select(**{s.alias: s.expr for s in selects})
    if explodes:
        for ex in explodes:
            q = q.explode(ex.list_attr, **ex.alias_to_expr)
    return q


def build_child_join_queries(trace_name: str,
                             child_kind: str,
                             selects_from_root: List[SelectSpec],
                             child_explodes: List[ExplodeSpec] | None) -> Tuple[SpanQuery, SpanQuery]:
    """
    Build two queries that can be joined by span_id:
      1) root spans filtered by name, select desired root-level fields
      2) child spans filtered by span_kind, with parent_id reindexed as span_id
         (so pandas can inner-join on span_id)
    """
    name_literal = json.dumps(trace_name)

    # Query #1: root spans (index: span_id)
    q_root = SpanQuery().where(f"parent_id is None and name == {name_literal}")
    if selects_from_root:
        q_root = q_root.select(**{s.alias: s.expr for s in selects_from_root})

    # Query #2: child spans by kind, reindex on parent_id => span_id
    q_child = SpanQuery().where(f"span_kind == {json.dumps(child_kind)}") \
                         .select(span_id="parent_id")  # special: sets index to parent span

    if child_explodes:
        for ex in child_explodes:
            q_child = q_child.explode(ex.list_attr, **ex.alias_to_expr)

    return q_root, q_child


def connect_client(base_url: Optional[str], api_key: Optional[str]) -> Client:
    """
    Construct a Phoenix Client. If base_url/api_key not provided, Client will
    read env vars:
      - PHOENIX_COLLECTOR_ENDPOINT (base URL)
      - PHOENIX_CLIENT_HEADERS (e.g., 'api_key=...')
    """
    headers = {}
    if api_key:
        # Phoenix Cloud expects 'api_key=...' in headers (client also accepts api_key param)
        headers["api_key"] = api_key
    return Client(base_url=base_url, api_key=api_key, headers=headers or None)


def write_dataframe(df: pd.DataFrame, out_path: str) -> None:
    out_path = out_path.strip()
    if out_path.lower().endswith(".parquet"):
        df.to_parquet(out_path, index=True)
    elif out_path.lower().endswith(".csv"):
        df.to_csv(out_path, index=True)
    else:
        raise ValueError("Output file must end with .parquet or .csv")
    logging.info("Wrote %d rows to %s", len(df), out_path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export parts of Phoenix traces by root span name")
    p.add_argument("--project", required=True, help="Phoenix project name or ID")
    p.add_argument("--trace-name", required=True, help="Root span name to match exactly")
    p.add_argument("--select", action="append", default=[],
                   help="alias=span_attr (repeatable), e.g. input=input.value")
    p.add_argument("--explode", action="append", default=[],
                   help="list_attr:alias=expr[,alias=expr...] (repeatable). "
                        "e.g. retrieval.documents:reference=document.content,score=document.score")
    p.add_argument("--child-kind", default=None,
                   help="Optional: pull from child spans with this span_kind (e.g., LLM, RETRIEVER, TOOL). "
                        "If provided, results are joined to matching roots by span_id.")
    p.add_argument("--start-time", default=None, help="Optional ISO-8601 lower bound")
    p.add_argument("--end-time", default=None, help="Optional ISO-8601 upper bound")
    p.add_argument("--limit", type=int, default=1000, help="Max rows to return (per query)")
    p.add_argument("--out", required=True, help="Output file (.parquet or .csv)")
    p.add_argument("--base-url", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
                   help="Phoenix endpoint (overrides env). Example: https://app.phoenix.arize.com")
    p.add_argument("--api-key", default=None, help="Phoenix API key (alternatively set PHOENIX_CLIENT_HEADERS)")

    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    selects = parse_select(args.select)
    explodes = parse_explode(args.explode) if args.explode else []

    if not selects and not explodes and not args.child_kind:
        # Provide a sensible default
        logging.info("No --select/--explode provided; defaulting to input/output.")
        selects = [SelectSpec("input", "input.value"), SelectSpec("output", "output.value")]

    client = connect_client(args.base_url, args.api_key)

    # Two modes:
    #  1) Root-only: filter by name and export attributes directly from the root span
    #  2) Child-join: export attributes from child spans (e.g., retriever docs) joined back to root
    start_time = args.start_time
    end_time = args.end_time

    if not args.child_kind:
        query = build_root_query(args.trace_name, selects, explodes)
        # Use the spandataframe API on the 'spans' surface (modern). Fallback: query_spans (legacy).
        try:
            df = client.spans.get_spans_dataframe(
                query=query,
                project_identifier=args.project,
                root_spans_only=True,
                start_time=start_time,
                end_time=end_time,
                limit=args.limit,
            )
        except Exception:
            # Legacy surface
            df = client.query_spans(
                query,
                project_name=args.project,
                root_spans_only=True,
                start_time=start_time,
                end_time=end_time,
                limit=args.limit,
            )
        if df is None or len(df) == 0:
            logging.warning("No matching spans found.")
            write_dataframe(pd.DataFrame(), args.out)
            return 0

        write_dataframe(df, args.out)
        return 0

    # Child-join mode
    q_root, q_child = build_child_join_queries(
        trace_name=args.trace_name,
        child_kind=args.child_kind,
        selects_from_root=selects,
        child_explodes=explodes,
    )

    try:
        dfs = client.spans.get_spans_dataframe(
            query=q_root, project_identifier=args.project,
            root_spans_only=True, start_time=start_time, end_time=end_time, limit=args.limit
        ), client.spans.get_spans_dataframe(
            query=q_child, project_identifier=args.project,
            root_spans_only=False, start_time=start_time, end_time=end_time, limit=args.limit
        )
    except Exception:
        dfs = client.query_spans(
            q_root, q_child, project_name=args.project,
            root_spans_only=None, start_time=start_time, end_time=end_time, limit=args.limit
        )
        # legacy returns list[dataframe]

    if not dfs or any(d is None or len(d) == 0 for d in dfs):
        logging.warning("No matching roots or child spans found.")
        write_dataframe(pd.DataFrame(), args.out)
        return 0

    # Join on span_id (child query was reindexed to parent as span_id)
    if isinstance(dfs, tuple):
        df_joined = pd.concat(dfs, axis=1, join="inner")
    else:
        df_joined = pd.concat(dfs, axis=1, join="inner")

    write_dataframe(df_joined, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

