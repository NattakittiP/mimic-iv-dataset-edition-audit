#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_demo_subset_of_full.py
=============================
Verifies (and quantifies) a fact that the "trustworthiness" comparison in
compare_trustworthiness.py leans on but never checks: MIMIC-IV Demo is
documented as a subsample of MIMIC-IV Full (the same ~100 patients' data,
also present in Full) -- NOT an independently drawn cohort. This matters for
how the Demo-vs-Full comparison should be framed: Demo isn't a
different population being compared to Full, it's a small, patient-clustered
slice of Full's own population.

This script reads subject_id directly from the raw release tables (not the
prepared analytic CSVs, to stay independent of any feature-engineering
choices) and reports the overlap. Works for both the Hospital module
(hosp/patients.csv.gz) and the ED module (ed/edstays.csv.gz).

Usage:
  python check_demo_subset_of_full.py \
      --demo_dir ../MIMIC_Dataset/MIMIC-IV-Demo-2.2 \
      --full_dir ../MIMIC_Dataset/MIMIC-IV-Full-2.2 \
      --module hosp --outdir ../Results/checks

  python check_demo_subset_of_full.py \
      --demo_dir ../MIMIC_Dataset/MIMIC-IV-ED-Demo-2.2 \
      --full_dir ../MIMIC_Dataset/MIMIC-IV-ED-Full-2.2 \
      --module ed --outdir ../Results/checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[check_demo_subset] {msg}")


def _read_table(root: Path, rel_path: str) -> pd.DataFrame:
    p = root / rel_path
    if not p.exists():
        p = root / rel_path.replace(".csv.gz", ".csv")
    if not p.exists():
        raise FileNotFoundError(f"Could not find {rel_path}(.gz) under {root}")
    return pd.read_csv(p, usecols=["subject_id"])


def load_subject_ids(root: Path, module: str) -> set:
    root = root.expanduser().resolve()
    if module == "hosp":
        df = _read_table(root, "hosp/patients.csv.gz")
    elif module == "ed":
        df = _read_table(root, "ed/edstays.csv.gz")
    else:
        raise ValueError(f"Unknown module: {module}")
    ids = set(pd.to_numeric(df["subject_id"], errors="coerce").dropna().astype(int).tolist())
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check/report subject_id overlap between a Demo release and its matching Full release.")
    p.add_argument("--demo_dir", type=str, required=True, help="Demo release root (e.g. .../MIMIC-IV-Demo-2.2 or .../MIMIC-IV-ED-Demo-2.2).")
    p.add_argument("--full_dir", type=str, required=True, help="Full release root.")
    p.add_argument("--module", type=str, required=True, choices=["hosp", "ed"])
    p.add_argument("--outdir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    demo_ids = load_subject_ids(Path(args.demo_dir), args.module)
    full_ids = load_subject_ids(Path(args.full_dir), args.module)

    overlap = demo_ids & full_ids
    demo_only = demo_ids - full_ids
    is_subset = len(demo_only) == 0

    result = {
        "module": args.module,
        "demo_dir": str(Path(args.demo_dir).resolve()),
        "full_dir": str(Path(args.full_dir).resolve()),
        "n_demo_patients": len(demo_ids),
        "n_full_patients": len(full_ids),
        "n_overlap": len(overlap),
        "n_demo_only": len(demo_only),
        "pct_demo_patients_in_full": (len(overlap) / len(demo_ids) * 100.0) if demo_ids else None,
        "demo_is_strict_subset_of_full": is_subset,
    }

    out_path = outdir / f"demo_subset_check_{args.module}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    log(f"Demo patients: {result['n_demo_patients']:,}")
    log(f"Full patients: {result['n_full_patients']:,}")
    log(f"Overlap: {result['n_overlap']:,} ({result['pct_demo_patients_in_full']:.2f}% of Demo's patients also appear in Full)")
    if is_subset:
        log("CONFIRMED: every Demo patient is present in Full (Demo is a strict subset, as documented by PhysioNet).")
    else:
        log(f"WARNING: {result['n_demo_only']:,} Demo patient(s) NOT found in the Full release you pointed at -- "
            f"double check --full_dir is the matching Full release (or Full release version differs from Demo's).")
    log(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
