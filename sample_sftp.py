
"""
parse_mcp_tools.py

Parse a Markdown document that lists MCP tools and extract each individual tool
with its contents (description, input/output schema, examples, and raw section markdown).

Features
- Robust Markdown AST via `markdown-it-py` (headings, paragraphs, fenced code)
- Heuristics to detect "tool sections" (no strict format required)
- Extracts: name, description, input_schema, output_schema, examples
- Exports to a single JSON file and/or splits into per-tool files
- Minimal dependencies: `markdown-it-py`, `pyyaml`

Usage
------
pip install markdown-it-py pyyaml

python parse_mcp_tools.py tools.md --out tools.json --split-dir ./tools_split

The script tries to be format-agnostic. It assumes each tool is documented under
a heading (e.g., "### search" or "## Tool: search") and that its details appear
until the next heading at the same or higher level.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies handling
try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
except Exception as e:
    raise SystemExit(
        "This script requires 'markdown-it-py'. Install it with:\n"
        "  pip install markdown-it-py\n"
        f"Import error: {e}"
    )

try:
    import yaml
except Exception as e:
    raise SystemExit(
        "This script requires 'pyyaml'. Install it with:\n"
        "  pip install pyyaml\n"
        f"Import error: {e}"
    )


@dataclass
class ExampleBlock:
    language: Optional[str]
    code: str
    title: Optional[str] = None


@dataclass
class ToolRecord:
    tool_name: str
    heading_level: int
    description: Optional[str]
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    examples: List[ExampleBlock]
    other_structured_blobs: List[Dict[str, Any]]
    raw_markdown: str


def _strip_jsonc_comments(text: str) -> str:
    """
    Remove // line comments and /* ... */ block comments for JSONC/TypeScript-like code blocks.
    Not a full parser; good enough for common docs.
    """
    # remove /* ... */ (including multiline)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    # remove // ... EOL
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def _try_parse_structured(code: str, lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Try to parse a code fence payload as JSON or YAML.
    - If lang hints JSON or YAML, we try accordingly.
    - For jsonc/ts/tsx we strip comments and try JSON first, then YAML.
    - If no lang, we heuristically try JSON then YAML.
    """
    text = code
    lang_l = (lang or "").lower()

    def parse_json(s: str) -> Optional[Dict[str, Any]]:
        import json as _json
        try:
            obj = _json.loads(s)
            if isinstance(obj, dict):
                return obj
            # accept top-level array for completeness, but prefer dicts for schemas
            if isinstance(obj, list):
                return {"_array": obj}
        except Exception:
            return None
        return None

    def parse_yaml(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = yaml.safe_load(s)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"_array": obj}
        except Exception:
            return None
        return None

    if lang_l in {"jsonc", "ts", "tsx", "typescript"}:
        text = _strip_jsonc_comments(text)
        return parse_json(text) or parse_yaml(text)

    if lang_l in {"json"}:
        return parse_json(text) or parse_yaml(text)

    if lang_l in {"yaml", "yml"}:
        return parse_yaml(text) or parse_json(text)

    # unknown language: try JSON then YAML
    return parse_json(text) or parse_yaml(text)


def _looks_like_json_schema(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    if d.get("type") == "object" and isinstance(d.get("properties"), dict):
        return True
    if "$schema" in d:
        return True
    # Some docs use "parameters" to mean input schema
    if isinstance(d.get("parameters"), dict):
        return True
    return False


def _section_text(lines: List[str], start_line: int, end_line: int) -> str:
    start_line = max(0, start_line)
    end_line = max(start_line, end_line)
    return "\n".join(lines[start_line:end_line]).rstrip() + "\n"


def _sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    return name or "tool"


def _extract_sections(md_tokens: List["Token"]) -> List[Tuple[int, str, int, int, int, int]]:
    """
    Return a list of sections:
        [(level, title, start_line, end_line, start_token_idx, end_token_idx), ...]
    end_line and end_token_idx are resolved in a second pass.
    """
    prelim = []
    for i, t in enumerate(md_tokens):
        if t.type == "heading_open" and t.tag and t.tag.startswith("h"):
            level = int(t.tag[1:])
            if i + 1 < len(md_tokens) and md_tokens[i + 1].type == "inline":
                title = md_tokens[i + 1].content.strip()
            else:
                title = "(untitled)"
            start_line = (t.map or [0, 0])[0]
            prelim.append([level, title, start_line, None, i, None])

    # resolve end_line and end_token_idx
    for idx in range(len(prelim)):
        level, title, start_line, _, start_tok_idx, _ = prelim[idx]
        # find the next section at same or higher level
        end_line = None
        end_tok_idx = None
        for j in range(idx + 1, len(prelim)):
            n_level, _, n_start_line, _, n_start_tok_idx, _ = prelim[j]
            if n_level <= level:
                end_line = n_start_line
                end_tok_idx = n_start_tok_idx
                break
        prelim[idx][3] = end_line  # may be None -> to be filled by caller
        prelim[idx][5] = end_tok_idx
    return [(a, b, c, d, e, f) for a, b, c, d, e, f in prelim]


def _classify_and_extract_from_section(
    tokens: List["Token"],
    lines: List[str],
    section: Tuple[int, str, int, Optional[int], int, Optional[int]],
) -> Optional[ToolRecord]:
    level, title, start_line, end_line, start_tok_idx, end_tok_idx = section

    # default end bounds if None
    if end_line is None:
        end_line = len(lines)
    if end_tok_idx is None:
        end_tok_idx = len(tokens)

    # Collect tokens inside the section (excluding the heading tokens themselves)
    # We start after the inline (title) and heading_close of this section.
    # Find precise bounds: from current heading_open idx to next heading_open/end
    # We'll just slice tokens from start_tok_idx+1 to end_tok_idx (exclusive).
    slice_tokens = tokens[start_tok_idx + 1 : end_tok_idx]

    # Extract description (first paragraph)
    description = None
    examples: List[ExampleBlock] = []
    input_schema = None
    output_schema = None
    other_blobs: List[dict] = []
    last_subheading_text = ""

    i = 0
    while i < len(slice_tokens):
        tok = slice_tokens[i]

        if tok.type == "heading_open":
            # capture subheading text (for classifying nearby code blocks)
            if i + 1 < len(slice_tokens) and slice_tokens[i + 1].type == "inline":
                last_subheading_text = slice_tokens[i + 1].content.strip().lower()
            else:
                last_subheading_text = ""
            i += 1
            continue

        if tok.type == "paragraph_open":
            # the following 'inline' contains paragraph text
            if i + 1 < len(slice_tokens) and slice_tokens[i + 1].type == "inline":
                para = slice_tokens[i + 1].content.strip()
                if para and description is None:
                    description = para
            i += 1
            continue

        if tok.type == "fence":
            lang = (tok.info or "").strip() or None
            code = tok.content

            # Try to interpret structured payload
            parsed = _try_parse_structured(code, lang)

            if parsed is not None:
                # Heuristics: if nearest subheading hints "input" or "params" -> input schema
                #             if hints "output" or "response" -> output schema
                #             else infer by JSON Schema shape
                sub = last_subheading_text

                if any(k in sub for k in ("input", "parameter", "parameters", "schema", "args", "request")):
                    if input_schema is None:
                        input_schema = parsed
                    else:
                        other_blobs.append(parsed)
                elif any(k in sub for k in ("output", "response", "result", "returns", "return")):
                    if output_schema is None:
                        output_schema = parsed
                    else:
                        other_blobs.append(parsed)
                else:
                    if _looks_like_json_schema(parsed):
                        if input_schema is None:
                            input_schema = parsed
                        else:
                            other_blobs.append(parsed)
                    else:
                        other_blobs.append(parsed)
            else:
                # Not structured -> treat as example
                examples.append(ExampleBlock(language=lang, code=code))

        i += 1

    # Decide whether this section is a "tool" section.
    # Heuristics: the section qualifies if any of these are true:
    # - title contains 'tool' (e.g., "Tool: search") OR
    # - we found an input_schema-like object OR
    # - we found at least one code example and the title looks like a function/tool name
    title_l = title.lower()
    looks_toolish_title = bool(re.search(r"\btool\b", title_l)) or bool(re.search(r"[a-z0-9_.-]{3,}", title_l))

    qualifies = False
    if input_schema is not None:
        qualifies = True
    elif "tool" in title_l:
        qualifies = True
    elif examples and looks_toolish_title:
        qualifies = True

    if not qualifies:
        return None

    raw = _section_text(lines, start_line, end_line)

    return ToolRecord(
        tool_name=title.strip(),
        heading_level=level,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        examples=examples,
        other_structured_blobs=other_blobs,
        raw_markdown=raw,
    )


def parse_mcp_tools(md_text: str, min_level: int = 2, max_level: int = 6) -> List[ToolRecord]:
    """
    Parse a Markdown document and return ToolRecords for each "tool-like" section.
    Parameters:
      - min_level: smallest heading level to consider (default: 2 -> '##')
      - max_level: largest heading level to consider (default: 6)
    """
    md = MarkdownIt()
    tokens: List[Token] = md.parse(md_text)
    lines = md_text.splitlines()

    secs = _extract_sections(tokens)
    records: List[ToolRecord] = []
    for sec in secs:
        level, title, start_line, end_line, start_tok_idx, end_tok_idx = sec
        if level < min_level or level > max_level:
            continue
        rec = _classify_and_extract_from_section(tokens, lines, sec)
        if rec is not None:
            records.append(rec)
    return records


def _write_split_dir(out_dir: str, tools: List[ToolRecord]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for t in tools:
        folder_name = _sanitize_filename(t.tool_name)
        tool_dir = os.path.join(out_dir, folder_name)
        os.makedirs(tool_dir, exist_ok=True)

        # Write raw markdown for the tool section
        with open(os.path.join(tool_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(t.raw_markdown)

        # Write description
        if t.description:
            with open(os.path.join(tool_dir, "DESCRIPTION.txt"), "w", encoding="utf-8") as f:
                f.write(t.description.strip() + "\n")

        # Write schemas
        if t.input_schema is not None:
            with open(os.path.join(tool_dir, "input_schema.json"), "w", encoding="utf-8") as f:
                json.dump(t.input_schema, f, indent=2, ensure_ascii=False)

        if t.output_schema is not None:
            with open(os.path.join(tool_dir, "output_schema.json"), "w", encoding="utf-8") as f:
                json.dump(t.output_schema, f, indent=2, ensure_ascii=False)

        # Write examples
        if t.examples:
            for idx, ex in enumerate(t.examples, 1):
                ext = (ex.language or "txt").split()[0].lower()
                path = os.path.join(tool_dir, f"example_{idx}.{ext}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(ex.code)

        # Other structured blobs (if any)
        if t.other_structured_blobs:
            with open(os.path.join(tool_dir, "other_structured.json"), "w", encoding="utf-8") as f:
                json.dump(t.other_structured_blobs, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse an MCP tools Markdown into structured JSON.")
    ap.add_argument("markdown_file", help="Path to the Markdown file to parse.")
    ap.add_argument("--out", "-o", help="Write all tools to a single JSON file.")
    ap.add_argument("--split-dir", help="Write each tool to its own subfolder under this directory.")
    ap.add_argument("--min-level", type=int, default=2, help="Minimum heading level to consider as a tool section (default: 2).")
    ap.add_argument("--max-level", type=int, default=6, help="Maximum heading level to consider as a tool section (default: 6).")
    args = ap.parse_args()

    with open(args.markdown_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    tools = parse_mcp_tools(md_text, min_level=args.min_level, max_level=args.max_level)

    if args.out:
        out_payload = [asdict(t) for t in tools]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

    if args.split_dir:
        _write_split_dir(args.split_dir, tools)

    # If neither out nor split-dir is provided, print a concise summary to stdout
    if not args.out and not args.split_dir:
        print(json.dumps([{
            "tool_name": t.tool_name,
            "has_input_schema": t.input_schema is not None,
            "has_output_schema": t.output_schema is not None,
            "examples": len(t.examples)
        } for t in tools], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

