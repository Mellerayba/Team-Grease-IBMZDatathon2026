"""Can our forecaster use the IBM Z AI accelerator? Tests every unknown and prints a plain verdict.

    python scripts/accel_check.py                                   # on the LinuxONE VM, in the notebook environment
    python scripts/accel_check.py --models notebooks/data/processed/forecast_models --rows 500000

Setup on the VM (once, in the notebook environment):   pip install --no-deps snapml      and      pip install nyoka lxml

Background (IBM's Snap ML docs): on IBM z16 and later, Snap ML can run INFERENCE of tree-ensemble models on the Integrated Accelerator for AI
through the zDNN library. It does not speed up training. Snap ML's importer only accepts PMML (a standard XML model format), so the route is
LightGBM model -> PMML (Nyoka, pure Python; checked to reproduce LightGBM exactly) -> snapml.import_model -> predict.
What this script settles, in order:
  1. does the CPU expose the accelerator (nnpa flag) and is the zDNN library installed?
  2. does a Snap ML wheel exist / import, and is Nyoka installed?
  3A. synthetic numeric Poisson model: does PMML import work, does Snap ML choose the accelerator ("zdnn_tensors"), do its outputs equal
      LightGBM's (the raw score, to which we apply exp for the Poisson mean), and is it faster?
  3B. our real model: it has categorical splits (zipcode, borough), which a numeric PMML cannot express, so it is reported, not converted.
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


def pip_probe():
    header("2. Snap ML and Nyoka")
    d = tempfile.mkdtemp()
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "snapml", "--no-deps", "--only-binary=:all:", "-d", d], capture_output=True, text=True, timeout=180)
        got = sorted(os.listdir(d))
        say(f"Snap ML wheel on PyPI for this machine: {got[0] if (r.returncode == 0 and got) else 'none (' + (r.stderr.strip().splitlines() or ['?'])[-1] + ')'}")
    except Exception as e:
        say(f"could not run the pip probe: {e}")


def load(name):
    try:
        mod = __import__(name)
        say(f"{name} {getattr(mod, '__version__', '')} imported")
        return mod
    except Exception as e:
        say(f"{name} not usable here ({type(e).__name__}: {str(e)[:110]})")
        return None


def has_categorical_splits(booster) -> bool:
    return any(s in str(booster.dump_model()["tree_info"]) for s in ('"decision_type": "=="', "'decision_type': '=='"))


def synthetic(n_features=20):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200_000, n_features)).astype("float32")
    y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]))
    return lgb.LGBMRegressor(objective="poisson", n_estimators=200, num_leaves=63, verbose=-1).fit(X, y), n_features


def export_pmml(nyoka, sk_model, n_features, path):
    from sklearn.pipeline import Pipeline
    fn = getattr(nyoka, "lgb_to_pmml", None) or getattr(nyoka, "lgbm_to_pmml", None)
    if fn is None:
        raise RuntimeError("this Nyoka version has no LightGBM exporter")
    fn(Pipeline([("model", sk_model)]), [f"f{i}" for i in range(n_features)], "y", str(path))


def with_algorithm_name(pmml_path, name):
    """Snap ML's importer requires algorithmName on the MiningModel element (it lists LightGBM / lightgbm / XGBoost ...); Nyoka does not write it."""
    text = Path(pmml_path).read_text()
    text = re.sub(r"<MiningModel([^>]*?)>", lambda m: m.group(0) if "algorithmName" in m.group(1) else f'<MiningModel{m.group(1)} algorithmName="{name}">', text, count=1)
    out = Path(str(pmml_path).replace(".pmml", f"_{name}.pmml"))
    out.write_text(text)
    return out


def pick_importable(snapml, pmml_path):
    """Try the algorithm names Snap ML's library lists for LightGBM; return the first PMML file it imports, or None."""
    for name in ("LightGBM", "lightgbm"):
        patched = with_algorithm_name(pmml_path, name)
        try:
            snapml.import_model(str(patched), input_type="pmml", tree_format="compress_trees")
            say(f"Snap ML accepts the PMML with algorithmName=\"{name}\"")
            return patched
        except Exception as e:
            say(f"algorithmName=\"{name}\": import FAILED: {type(e).__name__}: {str(e)[:300]}")
    return None


def snapml_test(snapml, pmml_path, booster, n_features, rows):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(rows, n_features)).astype("float32")
    t0 = time.time(); raw = booster.predict(X, raw_score=True); t_lgb = time.time() - t0
    say(f"LightGBM CPU: {rows:,} rows in {t_lgb:.2f}s ({rows / t_lgb / 1e6:.2f}M rows/s)")
    results = {}
    for fmt in ("auto", "zdnn_tensors", "compress_trees"):
        try:
            m = snapml.import_model(str(pmml_path), input_type="pmml", tree_format=fmt)
        except Exception as e:
            say(f"Snap ML import FAILED for tree_format={fmt}: {type(e).__name__}: {str(e)[:300]}")
            continue
        try:
            m.predict(X[:1000])                                       # warm-up
            t0 = time.time(); out = np.asarray(m.predict(X)).ravel(); dt = time.time() - t0
        except Exception as e:
            say(f"tree_format={fmt}: predict FAILED ({type(e).__name__}: {str(e)[:200]})"); continue
        rel = "raw score (apply exp for the Poisson mean)" if np.allclose(out, raw, atol=1e-3) else (
            "exp(raw) = Poisson mean" if np.allclose(out, np.exp(raw), rtol=1e-3, atol=1e-4) else "NEITHER (mismatch)")
        chosen = getattr(m, "optimized_tree_format_", "?")
        say(f"Snap ML tree_format={fmt:<14} chosen={chosen:<15} {rows / dt / 1e6:5.2f}M rows/s ({t_lgb / dt:4.1f}x LightGBM) | output equals: {rel} "
            f"| max diff {np.max(np.abs(out - raw)):.1e}")
        results[fmt] = (chosen, rel, t_lgb / dt)
    used = any(v[0] == "zdnn_tensors" for v in results.values())
    ok = bool(results) and all("NEITHER" not in v[1] for v in results.values())
    say(f"VERDICT: accelerator (zdnn_tensors) selected: {'YES' if used else 'NO'} | predictions match LightGBM: {'YES' if ok else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "notebooks/data/processed/forecast_models"), help="folder with our saved LightGBM models")
    ap.add_argument("--rows", type=int, default=500_000)
    args = ap.parse_args()
    hardware(); pip_probe()
    snapml, nyoka = load("snapml"), load("nyoka")
    if snapml is None or nyoka is None:
        say("\nSUMMARY: install the missing piece(s) in the notebook environment, then run this again:")
        if snapml is None: say("   pip install --no-deps snapml")
        if nyoka is None: say("   pip install nyoka lxml")
        return
    header("3A. Synthetic numeric Poisson model: does the accelerator path work at all?")
    d = Path(tempfile.mkdtemp())
    sk, nf = synthetic()
    try:
        export_pmml(nyoka, sk, nf, d / "synthetic.pmml")
        say(f"exported PMML ({(d / 'synthetic.pmml').stat().st_size / 1e6:.1f} MB)")
        ok_file = pick_importable(snapml, d / "synthetic.pmml")
        if ok_file is not None:
            snapml_test(snapml, ok_file, sk.booster_, nf, args.rows)
        else:
            say("VERDICT: Snap ML did not accept the exported PMML (see the errors above: each names the unsupported item and the line).")
    except Exception as e:
        say(f"PMML export or Snap ML test failed: {type(e).__name__}: {str(e)[:300]}")
    header("3B. Our real model (critical_1h)")
    real = Path(args.models) / "critical_1h.txt"
    if not real.exists():
        say(f"skipped: no saved models at {real}")
    else:
        import lightgbm as lgb
        b = lgb.Booster(model_file=str(real))
        if has_categorical_splits(b):
            say("our real model uses categorical splits (zipcode / borough), which a numeric PMML cannot express, so it cannot be converted as is.")
            say("fix if 3A works: retrain a numeric-only copy for the accelerator (zipcode and borough as numeric codes; lat/lon already carry location).")
        else:
            say("no categorical splits: could be converted the same way as 3A.")
    say("\nSUMMARY: send this whole output back (and a screenshot if 3A says YES / YES).")


if __name__ == "__main__":
    main()
