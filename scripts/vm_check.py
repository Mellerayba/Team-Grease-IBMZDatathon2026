"""Report what this machine offers and measure how much memory the data pipeline really needs here.

    python scripts/vm_check.py              # platform facts: CPU, memory limits, accelerator flags, installed libraries
    python scripts/vm_check.py --years 2    # also builds the panel for the latest 2 years of notebooks/ems_raw and reports peak memory and time

Run it on the LinuxONE VM (and screenshot the output: the judging rubric asks for evidence of platform use) and send us the output if a notebook kernel dies.
"""
import argparse
import importlib
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def gb(n_bytes) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def section(title: str) -> None:
    print(f"\n== {title}")


def platform_facts() -> None:
    section("Platform")
    print(f"machine        : {platform.machine()} ({sys.byteorder}-endian)")
    print(f"os             : {platform.platform()}")
    print(f"python         : {platform.python_version()}")
    print(f"logical cpus   : {os.cpu_count()}")
    cpu_max = read("/sys/fs/cgroup/cpu.max").split()
    if cpu_max and cpu_max[0] != "max":
        print(f"cpu limit      : {int(cpu_max[0]) / int(cpu_max[1]):.1f} cpus (container / cgroup limit)")
    sysinfo = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in read("/proc/sysinfo").splitlines() if ":" in l)}
    if sysinfo:                                             # exists on IBM Z only
        print(f"IBM Z machine  : manufacturer {sysinfo.get('Manufacturer')}, type {sysinfo.get('Type')}, model {sysinfo.get('Model')}")


def memory_facts() -> None:
    section("Memory (this is what decides whether a notebook kernel survives)")
    meminfo = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in read("/proc/meminfo").splitlines() if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal")}
    if meminfo:
        print(f"total          : {gb(meminfo.get('MemTotal', 0))}")
        print(f"available now  : {gb(meminfo.get('MemAvailable', 0))}   <- what is free right now, after other processes and old kernels")
        print(f"swap           : {gb(meminfo.get('SwapTotal', 0))}")
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = read(path).strip()
        if v and v != "max" and int(v) < 1 << 60:
            print(f"container limit: {gb(int(v))}   <- a cgroup limit can be far below the machine's total memory")
    print("tip            : if 'available now' is much lower than 'total', old kernels are holding memory:")
    print("                 shut them down (Jupyter: Kernel > Shut Down All Kernels) or run  ps aux --sort=-rss | head")


def accelerator_facts() -> None:
    section("CPU features and AI accelerator")
    features = ""
    for line in read("/proc/cpuinfo").splitlines():
        if line.lower().startswith(("features", "flags")):
            features = line.split(":", 1)[1].split()
            break
    if not features:
        print("no CPU feature list found here (not Linux?)")
        return
    interesting = {"vx": "vector facility (SIMD)", "vxe": "vector enhancements 1", "vxe2": "vector enhancements 2",
                   "nnpa": "neural network processing assist = the on-chip AI accelerator (IBM z16 / Telum and later)",
                   "sort": "sort instruction", "dflt": "deflate (compression) instruction"}
    for flag, meaning in interesting.items():
        print(f"{flag:<7}: {'YES' if flag in features else 'no ':<4} {meaning}")
    print("note           : the AI accelerator speeds up INFERENCE of neural networks and tree ensembles (via zDNN / Snap ML / ONNX).")
    print("                 It does not speed up LightGBM training or pandas data preparation.")


def library_facts() -> None:
    section("Libraries")
    for name in ("numpy", "pandas", "lightgbm", "sklearn", "snapml", "onnxruntime", "xgboost", "torch", "tensorflow"):
        try:
            mod = importlib.import_module(name)
            print(f"{name:<12}: {getattr(mod, '__version__', 'installed')}")
        except Exception as e:                                # an import can fail in many ways on a new architecture
            print(f"{name:<12}: not available ({type(e).__name__})")


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024     # Linux reports kilobytes, macOS bytes


def pipeline_memory(years: int) -> None:
    section(f"Pipeline memory test: panel + history features for the latest {years} year(s) in notebooks/ems_raw")
    sys.path.insert(0, str(ROOT / "src"))
    files = sorted((ROOT / "notebooks" / "ems_raw").glob("20??.csv"))[-years:]
    if not files:
        print("no notebooks/ems_raw/<year>.csv files found: run the fetch cell in HarryRowlands.ipynb first")
        return
    from grease.forecast import features as F
    base = peak_rss_bytes()
    t0 = time.time()
    panel, rows = F.build_panel_from_csvs(files)
    t1 = time.time()
    panel = F.add_history_features(panel)
    t2 = time.time()
    print(f"files          : {[f.name for f in files]} ({sum(rows.values()):,} incident rows)")
    print(f"panel          : {panel.shape[0]:,} rows x {panel.shape[1]} columns, {gb(panel.memory_usage().sum())} in memory")
    print(f"time           : {t1 - t0:.0f}s to build the panel, {t2 - t1:.0f}s for the history features")
    print(f"peak memory    : {gb(peak_rss_bytes())} (baseline before the test: {gb(base)})")
    print("compare with   : 'available now' above. If the peak is close to it, load fewer years (EMS_YEARS in the notebook settings).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=0, help="also run the pipeline memory test on the latest N years of EMS data")
    args = ap.parse_args()
    platform_facts(); memory_facts(); accelerator_facts(); library_facts()
    if args.years:
        pipeline_memory(args.years)
