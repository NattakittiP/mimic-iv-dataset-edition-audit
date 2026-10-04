#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_environment.py
=======================
Dumps the exact package versions and platform info actually used for a run,
to a JSON file. Run this once right after the final Full-scale run and keep
the output next to the results. Exact patch versions of numpy/pandas/
scikit-learn can shift RSCE numbers slightly (for example, pandas's
Copy-on-Write and string-dtype changes affect DataFrame handling), so
recording the environment that actually produced the numbers is what lets
someone else reproduce them exactly, and tells a numeric change caused by a
library version apart from a real change in the result.

Usage:
  python capture_environment.py --outdir Results/hosp_full/rsce
  # writes Results/hosp_full/rsce/environment_lock.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def get_pip_freeze() -> list[str]:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        return sorted(out.stdout.strip().splitlines())
    except Exception as e:
        return [f"ERROR: could not run pip freeze: {e}"]


def get_key_versions() -> dict:
    versions = {}
    for pkg in ["numpy", "pandas", "scipy", "matplotlib", "sklearn", "statsmodels", "shap", "tqdm", "tabulate"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "NOT INSTALLED"
    return versions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture exact package/platform versions for reproducibility.")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write environment_lock.json into (typically the same --outdir as the run you're capturing for).")
    p.add_argument("--note", type=str, default="", help="Optional free-text note (e.g. 'final Full hosp run').")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record = {
        "note": args.note,
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "key_package_versions": get_key_versions(),
        "pip_freeze": get_pip_freeze(),
    }

    out_path = outdir / "environment_lock.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[capture_environment] Wrote: {out_path.resolve()}")
    print(f"[capture_environment] Key versions: {json.dumps(record['key_package_versions'], indent=2)}")


if __name__ == "__main__":
    main()
