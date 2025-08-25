# pipeline_clean.py
# Run: python pipeline_clean.py --ids A B C --config pipeline.json --workers 4
from __future__ import annotations
import argparse, hashlib, json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

HASH_FILE = ".signature"
JSON_GLOB = "**/*.json"

# --------- tiny helpers ---------
def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def dir_json_hash(root: Path) -> str:
    """Stable hash of all .json files under root: filenames + contents."""
    h = hashlib.sha256()
    files = sorted(root.glob(JSON_GLOB))
    for f in files:
        h.update(str(f.relative_to(root)).encode("utf-8"))
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def stage_spec_hash(script: Path, args: List[str]) -> str:
    """Hash that captures the stage 'definition' = code + static args."""
    h = hashlib.sha256()
    h.update(file_hash(script).encode())
    for a in args:
        h.update(a.encode("utf-8"))
    return h.hexdigest()

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    tmp.replace(path)

def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None

def substitute(args: List[str], mapping: Dict[str, str]) -> List[str]:
    """Apply {placeholders} per-arg; leave unknown placeholders untouched."""
    class SafeMap(dict):
        def __missing__(self, key): return "{" + key + "}"
    m = SafeMap(mapping)
    return [a.format_map(m) for a in args]

# --------- core per-ID pipeline ---------
def process_one(id_: str, cfg: dict, force: bool) -> str:
    raw_root = Path(cfg.get("raw_root", "data/raw"))
    proc_root = Path(cfg.get("proc_root", "data/processed"))
    s1 = cfg["stage1"]; s2 = cfg["stage2"]

    raw_dir = raw_root / id_
    proc_dir = proc_root / id_
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: run generator with arbitrary extra args from config
    s1_script = Path(s1["script"])
    s1_args = substitute(s1.get("args", []), {
        "id": id_, "out_dir": str(raw_dir), "in_dir": str(raw_dir)  # in_dir provided just in case
    })
    run([sys.executable, str(s1_script), *s1_args])

    # --- Compute current signature (data + stage2 definition)
    raw_hash = dir_json_hash(raw_dir)
    s2_script = Path(s2["script"])
    s2_args = substitute(s2.get("args", []), {
        "id": id_, "in_dir": str(raw_dir), "out_dir": str(proc_dir)
    })
    spec_hash = stage_spec_hash(s2_script, s2_args)
    signature = hashlib.sha256((raw_hash + spec_hash).encode("utf-8")).hexdigest()

    # --- Skip if up-to-date (unless forced)
    old_sig = read_text(proc_dir / HASH_FILE)
    if not force and old_sig == signature:
        print(f"[{id_}] up-to-date; skip stage 2")
        return "skipped"

    # --- Stage 2: process
    run([sys.executable, str(s2_script), *s2_args])

    # Record signature atomically
    atomic_write_text(proc_dir / HASH_FILE, signature)
    print(f"[{id_}] processed")
    return "done"

# --------- CLI ---------
def load_config(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    # Optional TOML support on Python 3.11+: use tomllib if you prefer .toml
    try:
        import tomllib  # type: ignore
        if path.suffix.lower() in (".toml", ".tml"):
            return tomllib.loads(path.read_text())
    except Exception:
        pass
    raise SystemExit(f"Unsupported config type: {path.suffix}. Use .json or .toml (Py>=3.11).")

def main():
    ap = argparse.ArgumentParser(description="Clean two-stage pipeline (generator -> processor)")
    ap.add_argument("--ids", nargs="+", required=True, help="IDs to process")
    ap.add_argument("--config", required=True, help="Path to pipeline.json/.toml")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel IDs")
    ap.add_argument("--force", action="store_true", help="Force stage 2 even if signature matches")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, i, cfg, args.force): i for i in args.ids}
        for f in as_completed(futs):
            i = futs[f]
            try:
                f.result()
                ok += 1
            except subprocess.CalledProcessError as e:
                print(f"[{i}] FAILED (exit {e.returncode})")
                fail += 1
            except Exception as e:
                print(f"[{i}] FAILED ({e})")
                fail += 1
    print(f"Summary: {ok} ok, {fail} failed")

if __name__ == "__main__":
    main()

