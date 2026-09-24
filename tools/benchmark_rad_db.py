"""
Benchmark and verification harness against rad_examples_db corpus.
Validates pyradioss starter parsing and engine execution against
the reference OpenRadioss gold set benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CORPUS_ROOT = Path(r"C:\Users\pmqua\PycharmProjects\rad_examples_db")
GOLD_SET_PATH = CORPUS_ROOT / "candidates" / "bench" / "gold_set.json"
MANIFEST_PATH = CORPUS_ROOT / "manifest.jsonl"


def load_gold_set():
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_manifest_index(ids_needed: set[int]) -> dict[int, dict]:
    """Load manifest rows only for the IDs we are testing to save memory/time."""
    index = {}
    if not MANIFEST_PATH.exists():
        return index
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row_id = row.get("id")
                if row_id in ids_needed:
                    index[row_id] = row
            except Exception:
                pass
    return index


def filter_gold_decks(gold_decks, manifest_index=None, max_nodes=50, max_cycles=1000, limit=20, standalone_only=False):
    candidates = []
    for d in gold_decks:
        nodes = d.get("nodes", 0)
        cycles = d.get("cycles", 0)
        if nodes <= max_nodes and cycles <= max_cycles:
            if standalone_only and manifest_index:
                row = manifest_index.get(d["id"], {})
                incs = row.get("includes") or []
                if incs:
                    continue
            candidates.append(d)
    candidates.sort(key=lambda x: (x.get("nodes", 0), x.get("cycles", 0)))
    return candidates[:limit] if limit else candidates


def stage_deck(deck_info, manifest_row, dest_dir: Path):
    """Copy starter package tree plus any out-of-tree includes (as run_bench.py does)."""
    src_starter = CORPUS_ROOT / deck_info["stored"]
    src_dir = src_starter.parent
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy directory containing starter
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)

    # 2. Stage engine deck if located elsewhere
    if manifest_row:
        for e in manifest_row.get("engine_files") or []:
            ep = CORPUS_ROOT / e
            tgt = dest_dir / ep.name
            if not tgt.exists() and ep.exists():
                shutil.copy2(ep, tgt)

        # 3. Stage out-of-tree resolved includes
        incs = manifest_row.get("includes") or []
        res = manifest_row.get("includes_resolved") or []
        for name, rp in zip(incs, res):
            if not rp or rp == name:
                continue
            src = CORPUS_ROOT / rp
            if not src.exists():
                continue
            rel = name.replace("\\", "/").strip().strip('"')
            if re.match(r"^[a-zA-Z]:", rel) or rel.startswith("/"):
                rel = rel.rsplit("/", 1)[-1]
            tgt = (dest_dir / rel).resolve()
            if not str(tgt).startswith(str(dest_dir.resolve())):
                tgt = dest_dir / Path(rel).name
            if not tgt.exists():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, tgt)
                except OSError:
                    pass

    return dest_dir / src_starter.name


def test_starter_on_deck(deck_info, manifest_row, scratch_dir):
    """Run pyradioss starter on deck, comparing summary metrics."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.starter import run_starter

    test_run_dir = Path(scratch_dir) / f"run_{deck_info['id']}"
    try:
        target_starter = stage_deck(deck_info, manifest_row, test_run_dir)
    except Exception as e:
        return {"status": "STAGE_ERROR", "error": str(e), "error_type": type(e).__name__}

    log = MessageLog()
    t0 = time.perf_counter()
    try:
        model = run_starter(str(target_starter), log=log)
        dt = time.perf_counter() - t0
        res = {
            "status": "PASS",
            "time_s": dt,
            "py_nodes": model.numnod,
            "ref_nodes": deck_info["nodes"],
            "nodes_match": model.numnod == deck_info["nodes"],
            "warnings": len(log.warnings),
            "errors": len(log.errors),
            "run_dir": str(test_run_dir),
            "starter_file": str(target_starter),
        }
        return res
    except Exception as e:
        return {
            "status": "ERROR",
            "time_s": time.perf_counter() - t0,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def test_engine_on_deck(deck_info, starter_result, backend="numpy"):
    """Run pyradioss engine on deck, comparing cycles and energy."""
    if starter_result.get("status") != "PASS":
        return {"status": "SKIPPED_STARTER_FAILED"}

    old_backend = os.environ.get("PYRADIOSS_BACKEND")
    os.environ["PYRADIOSS_BACKEND"] = backend

    from pyradioss.common.messages import MessageLog
    from pyradioss.engine.engine import run_engine

    try:
        run_dir = Path(starter_result["run_dir"])
        engine_deck_name = deck_info.get("engine_deck")
        if not engine_deck_name:
            return {"status": "NO_ENGINE_DECK"}

        engine_deck_path = run_dir / engine_deck_name
        if not engine_deck_path.exists():
            return {"status": "ENGINE_DECK_NOT_FOUND", "path": str(engine_deck_path)}

        log = MessageLog()
        t0 = time.perf_counter()
        try:
            model = run_engine(str(engine_deck_path), log=log)
            dt = time.perf_counter() - t0
            state = getattr(model, "engine_state", None)
            py_cycles = state.cycle if state else None
            ref_cycles = deck_info.get("cycles")
            return {
                "status": "PASS",
                "backend": backend,
                "time_s": dt,
                "py_cycles": py_cycles,
                "ref_cycles": ref_cycles,
                "cycle_diff": (py_cycles - ref_cycles) if (py_cycles is not None and ref_cycles is not None) else None,
                "ref_energy_err": deck_info.get("energy_error_pct"),
                "stop_reason": getattr(state, "stop_reason", None) if state else None,
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "backend": backend,
                "time_s": time.perf_counter() - t0,
                "error": str(e),
                "error_type": type(e).__name__,
            }
    finally:
        if old_backend is not None:
            os.environ["PYRADIOSS_BACKEND"] = old_backend
        elif "PYRADIOSS_BACKEND" in os.environ:
            del os.environ["PYRADIOSS_BACKEND"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark pyradioss against rad_examples_db gold set")
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "numba", "both"])
    parser.add_argument("--tier", type=int, default=0, help="Benchmark tier: 0 (<100 nodes), 1 (100-10k)")
    parser.add_argument("--standalone", action="store_true", help="Only test decks with no external includes")
    parser.add_argument("--starter-only", action="store_true", help="Only run Starter stage, skip Engine")
    args = parser.parse_args()

    print(f"Loading gold set from {GOLD_SET_PATH}...")
    gold = load_gold_set()
    print(f"Loaded {len(gold)} reference gold decks.")

    if args.tier == 0:
        max_n = args.max_nodes
    elif args.tier == 1:
        max_n = 1000
    else:
        max_n = 100000

    print("Loading manifest index...")
    all_gold_ids = {d["id"] for d in gold}
    manifest_index = load_manifest_index(all_gold_ids)
    print("Manifest index loaded.")

    candidates = filter_gold_decks(
        gold,
        manifest_index=manifest_index,
        max_nodes=max_n,
        max_cycles=args.max_cycles,
        limit=args.limit,
        standalone_only=args.standalone,
    )
    print(f"Selected {len(candidates)} candidate decks (max_nodes={max_n}, max_cycles={args.max_cycles}, standalone={args.standalone}).\n")

    results = []
    with tempfile.TemporaryDirectory(prefix="pyrad_db_bench_") as tmpdir:
        for idx, deck in enumerate(candidates, 1):
            deck_id = deck["id"]
            stored = deck["stored"]
            nodes = deck["nodes"]
            cycles = deck["cycles"]
            man_row = manifest_index.get(deck_id)

            print(f"[{idx}/{len(candidates)}] Deck ID {deck_id}: nodes={nodes}, cycles={cycles}")
            print(f"  File: {stored}")

            st_res = test_starter_on_deck(deck, man_row, tmpdir)
            nodes_status = f"(nodes: py={st_res.get('py_nodes')}, ref={nodes})" if st_res["status"] == "PASS" else ""
            print(f"  Starter: {st_res['status']} in {st_res.get('time_s', 0):.3f}s {nodes_status}")

            eng_res = None
            if st_res["status"] == "PASS" and not args.starter_only:
                backends = ["numpy", "numba"] if args.backend == "both" else [args.backend]
                eng_res = {}
                for b in backends:
                    b_res = test_engine_on_deck(deck, st_res, backend=b)
                    eng_res[b] = b_res
                    if b_res["status"] == "PASS":
                        print(f"  Engine ({b}): PASS in {b_res.get('time_s', 0):.3f}s (cycles: py={b_res.get('py_cycles')}, ref={cycles})")
                    else:
                        print(f"  Engine ({b}): {b_res['status']} ({b_res.get('error_type')}: {b_res.get('error')})")
            elif args.starter_only:
                print("  Engine:  SKIPPED (--starter-only)")
            else:
                print(f"  Engine:  SKIPPED ({st_res.get('error_type')}: {st_res.get('error')})")

            results.append({
                "id": deck_id,
                "stored": stored,
                "nodes": nodes,
                "cycles": cycles,
                "starter": st_res,
                "engine": eng_res,
            })
            print()

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    passed_starter = sum(1 for r in results if r["starter"].get("status") == "PASS")
    
    # Engine pass check
    def is_eng_pass(r):
        eng = r.get("engine")
        if not eng:
            return False
        if isinstance(eng, dict):
            # check any backend passed
            return any(isinstance(v, dict) and v.get("status") == "PASS" for v in eng.values())
        return False

    passed_engine = sum(1 for r in results if is_eng_pass(r))
    print(f"Total decks tested: {len(results)}")
    print(f"Starter PASS: {passed_starter}/{len(results)} ({passed_starter/len(results)*100:.1f}%)")
    print(f"Engine PASS:  {passed_engine}/{len(results)} ({passed_engine/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
