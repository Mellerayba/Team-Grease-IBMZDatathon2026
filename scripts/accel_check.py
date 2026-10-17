"""Can our forecaster use the IBM Z AI accelerator? Tests every unknown and prints a plain verdict.

    python scripts/accel_check.py                                   # on the LinuxONE VM, in the notebook environment
    python scripts/accel_check.py --models notebooks/data/processed/forecast_models --rows 1000000

Background (IBM's Snap ML docs): on IBM z16 and later, Snap ML can run INFERENCE of tree-ensemble models (such as LightGBM) on the Integrated
Accelerator for AI through the zDNN library. It does not speed up training. Unknowns this script settles, in order:
  1. does the CPU expose the accelerator (nnpa flag) and is the zDNN library installed?
  2. does a Snap ML wheel exist for this machine and Python, and does it import?
  3. can it import a LightGBM Poisson model (own file formats vary by version), and do its predictions equal LightGBM's (raw score or exp of it)?
  4. does it pick the accelerator ("zdnn_tensors"), and is it any faster?
Test A uses a synthetic numeric model, so it separates "the accelerator works on this VM" from "our real model converts".
Test B uses our saved model, which has two categorical features (zipcode, borough) that many converters do not support.
"""
import argparse
import ctypes.util
import glob
import os
import platform
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
    nnpa = "nnpa" in feats
    say(f"AI accelerator flag (nnpa): {'YES' if nnpa else 'no / not an IBM Z machine'}")
    cands = [os.environ.get("SNAPML_ZDNN_LIB", ""), ctypes.util.find_library("zdnn") or ""]
    for pat in ("/usr/lib/s390x-linux-gnu/libzdnn*", "/usr/local/lib/libzdnn*", "/usr/lib64/libzdnn*", "/usr/lib/libzdnn*"):
        cands += glob.glob(pat)
    found = [c for c in cands if c]
    say(f"zDNN library: {found[0] if found else 'NOT FOUND (Snap ML needs libzdnn; try `apt-cache policy libzdnn1`, or set SNAPML_ZDNN_LIB=/path/libzdnn.so)'}")
    return nnpa, bool(found)


def pip_probe():
    header("2. Is there a Snap ML wheel for this machine?")
    d = tempfile.mkdtemp()
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "snapml", "--no-deps", "--only-binary=:all:", "-d", d],
                           capture_output=True, text=True, timeout=180)
        got = sorted(os.listdir(d))
        if r.returncode == 0 and got:
            say(f"wheel available: {got[0]}   -> install with: pip install snapml")
        else:
            tail = (r.stderr.strip().splitlines() or ["(no output)"])[-1]
            say(f"no usable wheel: {tail}")
            say("(the Snap ML docs list Python 3.7-3.9; this Python is " + platform.python_version() + ". A conda or venv with an older Python may be needed.)")
    except Exception as e:                                         # no internet, pip missing, timeout
        say(f"could not run the pip probe: {e}")


def import_snapml():
    try:
        import snapml
        say(f"snapml {getattr(snapml, '__version__', '?')} imported")
        return snapml
    except Exception as e:
        say(f"snapml not usable here ({type(e).__name__}: {str(e)[:120]})")
        return None


def synthetic_model(path, n_features=20):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200_000, n_features)).astype("float32")
    y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]))
    m = lgb.LGBMRegressor(objective="poisson", n_estimators=200, num_leaves=63, verbose=-1).fit(X, y)
    m.booster_.save_model(str(path))
    return lgb.Booster(model_file=str(path)), n_features


def try_import(snapml, path, booster, n_features, tree_format):
    """Try the file formats Snap ML might accept for LightGBM, return (model, how) or (None, reason)."""
    reasons = []
    for input_type in ("lightgbm", "text", "lgbm"):
        try:
            return snapml.import_model(str(path), input_type=input_type, tree_format=tree_format), f"input_type='{input_type}'"
        except Exception as e:
            reasons.append(f"{input_type}: {type(e).__name__} {str(e)[:70]}")
    try:                                                              # ONNX route, if the converter is installed
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        import lightgbm as lgb
        onnx_path = Path(tempfile.mkdtemp()) / "m.onnx"
        sk = lgb.LGBMRegressor(objective="poisson", n_estimators=1, verbose=-1)   # converter wants an sklearn wrapper
        sk._Booster = booster; sk._n_features = n_features; sk.fitted_ = True
        onx = onnxmltools.convert_lightgbm(sk, initial_types=[("input", FloatTensorType([None, n_features]))])
        onnx_path.write_bytes(onx.SerializeToString())
        return snapml.import_model(str(onnx_path), input_type="onnx", tree_format=tree_format), "via ONNX"
    except Exception as e:
        reasons.append(f"onnx: {type(e).__name__} {str(e)[:70]}")
    return None, "; ".join(reasons)


def compare_and_time(snapml, name, path, booster, n_features, rows, nnpa):
    header(f"{name}")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(rows, n_features)).astype("float32")
    t0 = time.time(); raw = booster.predict(X, raw_score=True); t_lgb = time.time() - t0
    say(f"LightGBM CPU: {rows:,} rows in {t_lgb:.2f}s ({rows / t_lgb / 1e6:.2f}M rows/s)")
    results = {}
    for fmt in ("auto", "zdnn_tensors", "compress_trees"):
        m, how = try_import(snapml, path, booster, n_features, fmt)
        if m is None:
            say(f"Snap ML import failed for tree_format={fmt}: {how}")
            if fmt == "auto":
                say("-> cannot import this model into Snap ML (so no accelerator for it)")
                return None
            continue
        try:
            t0 = time.time(); out = np.asarray(m.predict(X)).ravel(); dt = time.time() - t0
        except Exception as e:
            say(f"tree_format={fmt}: predict failed ({type(e).__name__}: {str(e)[:80]})"); continue
        rel = "raw score" if np.allclose(out, raw, atol=1e-3) else ("exp(raw) = Poisson mean" if np.allclose(out, np.exp(raw), rtol=1e-3, atol=1e-4) else "NEITHER (mismatch)")
        say(f"Snap ML tree_format={fmt:<14} chosen={getattr(m, 'optimized_tree_format_', '?'):<15} {rows / dt / 1e6:5.2f}M rows/s "
            f"({t_lgb / dt:4.1f}x LightGBM) | output equals: {rel} | max diff vs raw {np.max(np.abs(out - raw)):.2e} ({how})")
        results[fmt] = (getattr(m, "optimized_tree_format_", None), rel, t_lgb / dt)
    used = any(v[0] == "zdnn_tensors" for v in results.values())
    ok = any("MISMATCH" not in v[1].upper() for v in results.values())
    say(f"VERDICT: accelerator format selected: {'YES' if used else 'NO'} | predictions match LightGBM: {'YES' if ok and results else 'NO'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "notebooks/data/processed/forecast_models"), help="folder with our saved LightGBM models")
    ap.add_argument("--rows", type=int, default=500_000)
    args = ap.parse_args()
    nnpa, zdnn = hardware()
    pip_probe()
    header("2b. Import")
    snapml = import_snapml()
    if snapml is None:
        say("\nSUMMARY: Snap ML is not usable yet, so there is no accelerator path for tree models here. Fix the wheel / zDNN items above first.")
        say("Fallbacks: (1) run our Granite briefing on the VM through llama.cpp (IBM model, on IBM Z, fits in memory); (2) report the accelerator flag as available but unused.")
        return
    d = Path(tempfile.mkdtemp())
    booster, nf = synthetic_model(d / "synthetic_poisson.txt")
    a = compare_and_time(snapml, "3A. Synthetic numeric Poisson model (does the accelerator path work at all?)", d / "synthetic_poisson.txt", booster, nf, args.rows, nnpa)
    real = Path(args.models) / "critical_1h.txt"
    if real.exists():
        import lightgbm as lgb
        b = lgb.Booster(model_file=str(real))
        compare_and_time(snapml, "3B. Our real model (critical_1h; has categorical features zipcode and borough)", real, b, b.num_feature(), args.rows, nnpa)
    else:
        say(f"\n3B skipped: no saved models at {real}")
    say("\nSUMMARY: send this whole output back; if 3A says YES/YES but 3B fails, we retrain a numeric-only copy of the model (zipcode and borough as numbers) for the accelerator.")


if __name__ == "__main__":
    main()
