#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_compare_base.py
=====================
compare_pro.py / compare_addons.py / compare_world_heatmap.py expect a single
folder containing paired `demo_<suffix>` / `full_<suffix>` files (CSV and,
optionally, schema.json). run_rsce.py's own --outdir does not prefix its
output files that way, so this small helper copies the two run_rsce.py output
folders (Demo and Full) into one folder with the right prefixes, instead of
doing it by hand.

Usage:
  python make_compare_base.py \
      --demo_dir ../Results/hosp_demo/rsce \
      --full_dir ../Results/hosp_full/rsce \
      --outdir   ../Results/compare/hosp/_base
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

COPY_EXTS = (".csv", ".json")


def copy_prefixed(src_dir: Path, prefix: str, outdir: Path) -> int:
    n = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in COPY_EXTS:
            shutil.copy2(f, outdir / f"{prefix}{f.name}")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a demo_/full_ prefixed comparison base folder from two run_rsce.py --outdir folders.")
    p.add_argument("--demo_dir", type=str, required=True)
    p.add_argument("--full_dir", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--demo_prefix", type=str, default="demo_")
    p.add_argument("--full_prefix", type=str, default="full_")
    return p.parse_args()


REQUIRED = ("rsce_scores.csv", "metrics_aggregated.csv", "metrics_per_fold.csv", "ablation_summary.csv",
            "ablation_components_per_fold.csv", "rsce_per_fold.csv", "paired_tests.csv", "schema.json")


def check_run_dir(d: Path, label: str) -> dict:
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{label} run folder {d} is missing {missing} -- is it a finished run_rsce.py output "
                                f"(current version), not an --estimate_only probe?")
    schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    if schema.get("estimate_only"):
        raise ValueError(f"{label} run folder {d} is an --estimate_only probe (2 folds x 1 repeat), not a real run.")
    return schema


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    sd = check_run_dir(Path(args.demo_dir), "Demo")
    sf = check_run_dir(Path(args.full_dir), "Full")
    for k in ("folds", "repeats", "seed", "shap_samples", "svc_max_train_n", "compute_shap", "cv_mode", "worlds", "scoring",
              "model_params"):
        if sd.get(k) != sf.get(k):
            raise ValueError(f"Demo and Full runs differ in '{k}': {sd.get(k)} vs {sf.get(k)} -- not comparable.")
    if sd.get("package_versions") != sf.get("package_versions"):
        print(f"[make_compare_base] WARNING: package versions differ between Demo and Full: "
              f"{sd.get('package_versions')} vs {sf.get('package_versions')}")
    outdir.mkdir(parents=True, exist_ok=True)
    stale = [f for f in outdir.iterdir() if f.is_file() and f.name.startswith((args.demo_prefix, args.full_prefix))]
    for f in stale:  # never mix files from an earlier pairing
        f.unlink()

    n_demo = copy_prefixed(Path(args.demo_dir), args.demo_prefix, outdir)
    n_full = copy_prefixed(Path(args.full_dir), args.full_prefix, outdir)

    print(f"[make_compare_base] Copied {n_demo} demo files and {n_full} full files into: {outdir.resolve()}")
    print("[make_compare_base] Point compare_pro.py / compare_addons.py / compare_world_heatmap.py --base at this folder.")


if __name__ == "__main__":
    main()
