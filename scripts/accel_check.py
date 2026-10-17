"""Can our forecaster use the IBM Z AI accelerator? Tests every unknown and prints a plain verdict.

    python scripts/accel_check.py                                   # on the LinuxONE VM, in the notebook environment
    python scripts/accel_check.py --models notebooks/data/processed/forecast_models --rows 500000

Setup on the VM (once, in the notebook environment):   pip install --no-deps snapml      (and, only for the PMML fallback:  pip install nyoka lxml)

Background (IBM's Snap ML docs and code): on IBM z16 and later, Snap ML can run INFERENCE of tree-ensemble models on the Integrated Accelerator
for AI through the zDNN library. It does not speed up training. The class-level importer
    snapml.BoostingMachineRegressor(n_jobs=os.cpu_count() or 1).import_model(file, "lightgbm", tree_format=...)
reads LightGBM's own text model file directly (the generic snapml.import_model only accepts PMML). Unsupported items raise an error that names
the item and the line. What this script settles, in order:
  1. does the CPU expose the accelerator (nnpa flag) and is the zDNN library installed?
  2. is Snap ML installed?
  3A. synthetic numeric Poisson model: does it import, does it choose the accelerator ("zdnn_tensors"), do its outputs equal LightGBM's
      (the raw score, to which we apply exp for the Poisson mean), and is it faster?
  3B. the same for our real saved model (it has categorical features zipcode and borough).
  If the direct route fails for the synthetic model, the PMML route (LightGBM -> Nyoka -> PMML) is tried as a fallback.
"""
import argparse
import ctypes.util
import glob
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
say = lambda *a: print(*a, flush=True)


def header(t):
    say(f"\n== {t}")


def hardware():
    header("1. Hardware and zDNN library")
    say(f"machine: {platform.machine()} | python {platform.python_version()}")
    feats = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("features"):
                feats = line.split(":", 1)[1].split()
                break
    except OSError:
        pass
    say(f"AI accelerator flag (nnpa): {'YES' if 'nnpa' in feats else 'no / not an IBM Z machine'}")
    cands = [os.environ.get("SNAPML_ZDNN_LIB", ""), ctypes.util.find_library("zdnn") or ""]
    for pat in ("/usr/lib/s390x-linux-gnu/libzdnn*", "/usr/local/lib/libzdnn*", "/usr/lib64/libzdnn*", "/usr/lib/libzdnn*"):
        cands += glob.glob(pat)
    found = [c for c in cands if c]
    say(f"zDNN library: {found[0] if found else 'NOT FOUND (Snap ML needs libzdnn; set SNAPML_ZDNN_LIB=/path/libzdnn.so)'}")


def load(name, required=True):
    try:
        mod = __import__(name)
        say(f"{name} {getattr(mod, '__version__', '')} imported")
        return mod
    except Exception as e:
        say(f"{name} not usable here ({type(e).__name__}: {str(e)[:110]})" + ("" if required else "  (only needed for the PMML fallback)"))
        return None


def make_inputs(model_txt, rows, seed=1):
    """Random inputs drawn from each feature's real range in the model file; categorical features get valid category codes."""
    infos = re.search(r"^feature_infos=(.*)$", Path(model_txt).read_text(), re.M).group(1).split()
    rng = np.random.default_rng(seed)
    X = np.zeros((rows, len(infos)), dtype=np.float32)
    cat = []
    for j, tok in enumerate(infos):
        if tok == "none":
            continue
        if tok.startswith("["):
            lo, hi = map(float, tok.strip("[]").split(":"))
            X[:, j] = rng.uniform(lo, hi, rows)
        else:
            cat.append(j)
            X[:, j] = rng.choice([float(c) for c in tok.split(":")], rows)
    return X, cat


def has_categorical_splits(booster) -> bool:
    return any(s in str(booster.dump_model()["tree_info"]) for s in ('"decision_type": "=="', "'decision_type': '=='"))


def synthetic(n_features=20):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200_000, n_features)).astype("float32")
    y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]))
    return lgb.LGBMRegressor(objective="poisson", n_estimators=200, num_leaves=63, verbose=-1).fit(X, y)


def compare(snapml, make_model, booster, X, label):
    """make_model(tree_format) -> imported Snap ML model. Prints each tree format's result; returns True if the accelerator format ran and matched."""
    t0 = time.time(); raw = booster.predict(X, raw_score=True); t_lgb = time.time() - t0
    say(f"LightGBM CPU: {len(X):,} rows in {t_lgb:.2f}s ({len(X) / t_lgb / 1e6:.2f}M rows/s)")
    results = {}
    for fmt in ("auto", "zdnn_tensors", "compress_trees"):
        try:
            m = make_model(fmt)
        except Exception as e:
            say(f"[{label}] import FAILED for tree_format={fmt}: {type(e).__name__}: {str(e)[:400]}")
            continue
        try:
            m.predict(X[:1000])                                                   # warm-up
            t0 = time.time(); out = np.asarray(m.predict(X)).ravel(); dt = time.time() - t0
        except Exception as e:
            say(f"[{label}] predict FAILED for tree_format={fmt}: {type(e).__name__}: {str(e)[:300]}"); continue
        rel = ("raw score (apply exp for the Poisson mean)" if np.allclose(out, raw, rtol=1e-3, atol=1e-3) else
               "exp(raw) = Poisson mean" if np.allclose(out, np.exp(raw), rtol=1e-3, atol=1e-4) else "NEITHER (mismatch)")
        chosen = getattr(getattr(m, "booster_", m), "optimized_tree_format_", "?")
        say(f"[{label}] tree_format={fmt:<14} chosen={chosen:<15} {len(X) / dt / 1e6:5.2f}M rows/s ({t_lgb / dt:5.1f}x LightGBM) | output equals: {rel} | max diff {np.max(np.abs(out - raw)):.1e}")
        results[fmt] = (chosen, rel)
    used = any(v[0] == "zdnn_tensors" for v in results.values())
    ok = bool(results) and all("NEITHER" not in v[1] for v in results.values())
    say(f"VERDICT [{label}]: accelerator (zdnn_tensors) selected: {'YES' if used else 'NO'} | predictions match LightGBM: {'YES' if ok else 'NO'}")
    return used and ok


def direct_route(snapml, txt):
    def make(fmt):
        m = snapml.BoostingMachineRegressor(n_jobs=os.cpu_count() or 1)
        m.import_model(str(txt), "lightgbm", tree_format=fmt)
        return m
    return make


def pmml_route(snapml, nyoka, sk_model, n_features, d):
    from sklearn.pipeline import Pipeline
    fn = getattr(nyoka, "lgb_to_pmml", None) or getattr(nyoka, "lgbm_to_pmml")
    base = d / "synthetic.pmml"
    fn(Pipeline([("model", sk_model)]), [f"f{i}" for i in range(n_features)], "y", str(base))
    text = re.sub(r"<MiningModel([^>]*?)>", lambda m: m.group(0) if "algorithmName" in m.group(1) else f'<MiningModel{m.group(1)} algorithmName="LightGBM">', base.read_text(), count=1)
    out = d / "synthetic_LightGBM.pmml"; out.write_text(text)

    def make(fmt):
        m = snapml.BoostingMachineRegressor(n_jobs=os.cpu_count() or 1)
        m.import_model(str(out), "pmml", tree_format=fmt)
        return m
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "notebooks/data/processed/forecast_models"), help="folder with our saved LightGBM models")
    ap.add_argument("--rows", type=int, default=500_000)
    args = ap.parse_args()
    hardware()
    header("2. Snap ML")
    snapml = load("snapml")
    if snapml is None:
        say("\nSUMMARY: install Snap ML in the notebook environment, then run this again:   pip install --no-deps snapml")
        return
    import lightgbm as lgb
    d = Path(tempfile.mkdtemp())

    header("3A. Synthetic numeric Poisson model: does the accelerator path work at all?")
    sk = synthetic()
    txt = d / "synthetic.txt"; sk.booster_.save_model(str(txt))
    X, _ = make_inputs(txt, args.rows)
    ok = compare(snapml, direct_route(snapml, txt), sk.booster_, X, "direct lightgbm import")
    if not ok:
        nyoka = load("nyoka", required=False)
        if nyoka is not None:
            say("trying the PMML fallback (LightGBM -> Nyoka -> PMML -> Snap ML):")
            try:
                compare(snapml, pmml_route(snapml, nyoka, sk, X.shape[1], d), sk.booster_, X, "pmml fallback")
            except Exception as e:
                say(f"PMML fallback failed: {type(e).__name__}: {str(e)[:300]}")

    header("3B. Our real model (critical_1h; has categorical features zipcode and borough)")
    real = Path(args.models) / "critical_1h.txt"
    if not real.exists():
        say(f"skipped: no saved models at {real}")
    else:
        b = lgb.Booster(model_file=str(real))
        say(f"{b.num_feature()} features, {b.num_trees()} trees, categorical splits: {'YES' if has_categorical_splits(b) else 'no'}")
        Xr, cat = make_inputs(real, args.rows)
        say(f"test inputs: realistic ranges; categorical columns {cat} get valid category codes")
        compare(snapml, direct_route(snapml, real), b, Xr, "our model")
    say("\nSUMMARY: send this whole output back (and a screenshot if a VERDICT line says YES / YES).")


if __name__ == "__main__":
    main()
