"""Benchmark sweep: when (if ever) does the IBM Z AI accelerator beat plain LightGBM for tree-ensemble inference?

    python scripts/accel_bench.py                       # on the LinuxONE VM, in the notebook environment (needs: pip install --no-deps snapml)
    python scripts/accel_bench.py --quick               # smaller grid, about 2 minutes

Fair comparison: Snap ML defaults to ONE thread while LightGBM defaults to all CPUs, so every engine is run with 1 thread and with all CPUs.
Engines:  LightGBM CPU  |  Snap ML compress_trees (its CPU path)  |  Snap ML zdnn_tensors (the AI accelerator path).
Grid: tree size (num_leaves) x number of trees x batch size. Batch 256 is a live cycle (we score about 235 ZIPs per 30-minute cycle); large batches
are bulk scoring (a whole test year is about 4 million rows). Each model is also checked for correct output (max difference vs LightGBM).
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

CPUS = os.cpu_count() or 1
say = lambda *a: print(*a, flush=True)


def train(leaves, trees, n_features=20):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100_000, n_features)).astype("float32")
    y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2] * X[:, 3]))
    return lgb.LGBMRegressor(objective="poisson", n_estimators=trees, num_leaves=leaves, verbose=-1, n_jobs=CPUS).fit(X, y).booster_


def timed(predict, X, batch):
    """Throughput (rows/s) and per-call latency (ms) when X is scored in chunks of `batch` rows."""
    predict(X[:min(batch, len(X))])                                             # warm-up
    calls = 0; t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        predict(X[i:i + batch]); calls += 1
    dt = time.perf_counter() - t0
    return len(X) / dt, dt / calls * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rows", type=int, default=100_000, help="rows scored per measurement")
    args = ap.parse_args()
    try:
        import snapml
    except Exception as e:
        say(f"snapml is not usable here ({type(e).__name__}: {e}). Install: pip install --no-deps snapml"); return
    say(f"CPUs: {CPUS} | snapml {getattr(snapml, '__version__', '?')}")

    grid = [(15, 100), (63, 100)] if args.quick else [(7, 100), (15, 100), (31, 100), (63, 100), (15, 400)]
    batches = [256, 4096, 100_000]
    rng = np.random.default_rng(1)
    Xall = rng.normal(size=(args.rows, 20)).astype("float32")
    d = Path(tempfile.mkdtemp())
    wins = []

    for leaves, trees in grid:
        b = train(leaves, trees)
        txt = d / f"m_{leaves}_{trees}.txt"; b.save_model(str(txt))
        raw = b.predict(Xall[:20000], raw_score=True)
        engines = {}
        for nt in sorted({1, CPUS}):
            engines[f"LightGBM CPU        {nt}t"] = (lambda X, nt=nt: b.predict(X, raw_score=True, num_threads=nt))
            for fmt, label in (("compress_trees", "Snap ML CPU path    "), ("zdnn_tensors", "Snap ML ACCELERATOR")):
                try:
                    m = snapml.BoostingMachineRegressor(n_jobs=nt)
                    m.import_model(str(txt), "lightgbm", tree_format=fmt)
                    chosen = getattr(m.booster_, "optimized_tree_format_", "?")
                    if chosen != fmt:
                        say(f"  note: asked for {fmt} but Snap ML chose {chosen}")
                    engines[f"{label} {nt}t"] = m.predict
                    if nt == CPUS and fmt == "zdnn_tensors":
                        say(f"\n=== leaves={leaves} trees={trees} | accelerator output max diff vs LightGBM: {np.max(np.abs(np.asarray(m.predict(Xall[:20000])).ravel() - raw)):.1e}")
                except Exception as e:
                    say(f"  {label} {nt}t: import failed ({type(e).__name__}: {str(e)[:120]})")
        say(f"{'engine':<28}" + "".join(f"{'batch ' + str(bt):>20}" for bt in batches) + "   (thousand rows/s | ms per call)")
        res = {}
        for name, fn in engines.items():
            row = f"{name:<28}"
            for bt in batches:
                tput, ms = timed(fn, Xall, bt)
                res[(name, bt)] = tput
                row += f"{tput / 1e3:>12.0f} | {ms:>6.1f}"
            say(row)
        for bt in batches:
            lgb_best = max(v for (n, b_), v in res.items() if n.startswith("LightGBM") and b_ == bt)
            acc = {n: v for (n, b_), v in res.items() if n.startswith("Snap ML ACC") and b_ == bt}
            if acc:
                n, v = max(acc.items(), key=lambda kv: kv[1])
                wins.append((leaves, trees, bt, v / lgb_best, n.strip()))

    say("\n== SUMMARY: accelerator throughput divided by the best LightGBM throughput (above 1.0 = the accelerator wins)")
    say(f"{'leaves':>7}{'trees':>7}" + "".join(f"{'batch ' + str(bt):>14}" for bt in batches))
    for leaves, trees in grid:
        r = {bt: next((w for l, t, b_, w, _ in wins if (l, t, b_) == (leaves, trees, bt)), float('nan')) for bt in batches}
        say(f"{leaves:>7}{trees:>7}" + "".join(f"{r[bt]:>13.2f}x" for bt in batches))
    best = max(wins, key=lambda w: w[3]) if wins else None
    if best:
        say(f"\nBest case: leaves={best[0]}, trees={best[1]}, batch={best[2]}: accelerator is {best[3]:.2f}x LightGBM ({'a win' if best[3] > 1 else 'still slower'}).")
    say("Send this whole output back.")


if __name__ == "__main__":
    main()
