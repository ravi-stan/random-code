from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple
import random, json

def sample_by_tool(
    records: List[Dict[str, Any]],
    per_tool: int = 1,
    seed: Optional[int] = None,
    *,
    selector: Optional[Callable[[Any], Iterable[Any]]] = None,
    unique: bool = True,
    with_replacement: bool = False,
    keep_source: bool = True,
    tools_filter: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate outputs across multiple dicts (records) keyed by tool name,
    optionally extract inner items (via selector), then sample candidates
    per tool and return a final dict.

    Parameters
    ----------
    records : list of dict
        Each dict maps `tool_name -> output` (any JSON-like object).
    per_tool : int
        Number of samples to return per tool. If 1, store a single object.
    seed : int | None
        RNG seed for reproducibility.
    selector : callable | None
        If provided, called as `selector(output)` to yield sample candidates
        from a tool’s output (useful when output is nested, e.g., {"items":[...]}).
        If None, the entire output is treated as one candidate.
    unique : bool
        Deduplicate candidates (by JSON-serialized representation).
    with_replacement : bool
        If True, sample with replacement; otherwise without.
    keep_source : bool
        If True, include the record index where the item came from.
    tools_filter : iterable[str] | None
        If provided, only process these tool names.

    Returns
    -------
    Dict[str, Any]
        Mapping tool -> sampled item (or list of items if per_tool > 1).
        If keep_source=True, each sampled item is a dict:
        {"value": <object>, "source_index": <int>}.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, Any]]] = {}

    def stable_key(obj: Any) -> str:
        try:
            return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            # Fallback for non-JSON-serializable objects
            return repr(obj)

    for rec_idx, rec in enumerate(records):
        for tool, output in rec.items():
            if tools_filter and tool not in tools_filter:
                continue
            items = list(selector(output)) if selector else [output]
            if not items:
                continue
            lst = buckets.setdefault(tool, [])
            if unique:
                seen = {stable_key(v) for _, v in lst}
                for v in items:
                    sk = stable_key(v)
                    if sk not in seen:
                        lst.append((rec_idx, v))
                        seen.add(sk)
            else:
                lst.extend((rec_idx, v) for v in items)

    result: Dict[str, Any] = {}
    for tool, candidates in buckets.items():
        if not candidates:
            continue
        if with_replacement:
            chosen = [rng.choice(candidates) for _ in range(per_tool)]
        else:
            k = min(per_tool, len(candidates))
            chosen = rng.sample(candidates, k)

        packaged = (
            [{"value": v, "source_index": i} for i, v in chosen]
            if keep_source else
            [v for i, v in chosen]
        )
        result[tool] = packaged[0] if per_tool == 1 else packaged

    return result

