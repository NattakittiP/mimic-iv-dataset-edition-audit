#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_table1.py
==============
Builds a combined "Table 1" (baseline characteristics) across however many of
the four dataset variants (hosp_demo, hosp_full, ed_demo, ed_full) you've
already run prepare_hosp.py / prepare_ed.py for. Meant to answer, in one
place, the first descriptive question about the two releases: how similar
ARE Demo and Full, at the raw cohort level, before any modeling happens?

For each dataset it reports: row count, unique-patient count, mean rows per
patient (the clustering that motivates --group_col / --cv_mode group in
run_rsce.py / run_ppv.py / compare_trustworthiness.py), label prevalence,
overall missingness, and -- when present -- age (mean/sd) and sex
distribution. If a `<csv>.cohort_flow.json` file (written by prepare_hosp.py
/ prepare_ed.py) sits next to a dataset's CSV, its cohort-flow counts
(inclusion/exclusion at each filtering step) are merged in automatically for
a CONSORT-style flow diagram.

Usage (point at whichever datasets you've already built -- 1 to 4 of them):
  python make_table1.py \
    --dataset hosp_demo=../Results/hosp_demo/demo_analytic_dataset_mortality_all_admissions.csv:label_mortality \
    --dataset hosp_full=../Results/hosp_full/full_analytic_dataset_mortality_all_admissions.csv:label_mortality \
    --dataset ed_demo=../Results/ed_demo/demo_ed_analytic_dataset_admission.csv:label_ed_admit \
    --dataset ed_full=../Results/ed_full/full_ed_analytic_dataset_admission.csv:label_ed_admit \
    --outdir ../Results/table1

Each --dataset is TAG=CSV_PATH:TARGET_COL. Repeat the flag for each dataset
you want included (order doesn't matter, and you don't need all four).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[make_table1] {msg}")


def parse_dataset_spec(spec: str) -> Dict[str, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"--dataset must look like TAG=CSV_PATH:TARGET_COL, got: {spec}")
    tag, rest = spec.split("=", 1)
    csv_path, target_col = rest.rsplit(":", 1)
    return {"tag": tag.strip(), "csv_path": csv_path.strip(), "target_col": target_col.strip()}


def find_cohort_flow(csv_path: Path) -> Optional[dict]:
    flow_path = csv_path.with_suffix("").with_name(csv_path.stem + ".cohort_flow.json")
    if flow_path.exists():
        try:
            return json.loads(flow_path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"WARNING: could not parse {flow_path}: {e}")
    return None


def describe_one(tag: str, csv_path: str, target_col: str) -> Dict[str, object]:
    p = Path(csv_path)
    if not p.exists():
        log(f"WARNING: {tag}: file not found ({p}); skipping.")
        return {"tag": tag, "status": "MISSING", "csv_path": str(p)}

    df = pd.read_csv(p)
    row: Dict[str, object] = {"tag": tag, "status": "OK", "csv_path": str(p), "n_rows": int(len(df)), "n_cols": int(df.shape[1])}

    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        row["target_col"] = target_col
        row["n_label_missing"] = int(y.isna().sum())
        yv = y.dropna()
        row["label_prevalence"] = float(yv.mean()) if len(yv) else np.nan
        row["n_label_positive"] = int((yv == 1).sum())
        row["n_label_negative"] = int((yv == 0).sum())
    else:
        log(f"WARNING: {tag}: target_col '{target_col}' not found in {p.name}.")

    if "subject_id" in df.columns:
        n_subj = int(df["subject_id"].nunique())
        row["n_unique_patients"] = n_subj
        row["mean_rows_per_patient"] = float(len(df) / n_subj) if n_subj else np.nan
    else:
        row["n_unique_patients"] = None
        row["mean_rows_per_patient"] = None

    age_col = next((c for c in ["anchor_age"] if c in df.columns), None)
    if age_col:
        a = pd.to_numeric(df[age_col], errors="coerce").dropna()
        row["age_mean"] = float(a.mean()) if len(a) else np.nan
        row["age_sd"] = float(a.std()) if len(a) else np.nan

    if "gender" in df.columns:
        vc = df["gender"].value_counts(normalize=True, dropna=True)
        for k, v in vc.items():
            row[f"pct_gender_{k}"] = float(v)

    row["overall_missing_rate"] = float(df.isna().mean().mean())

    flow = find_cohort_flow(p)
    if flow:
        for k, v in flow.items():
            if isinstance(v, (int, float, str)) or v is None:
                row[f"flow.{k}"] = v
    return row


def to_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined Table 1 / cohort-flow summary across dataset variants.")
    p.add_argument("--dataset", action="append", required=True, dest="datasets",
                    help="TAG=CSV_PATH:TARGET_COL. Repeat for each dataset variant to include.")
    p.add_argument("--outdir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = [parse_dataset_spec(s) for s in args.datasets]
    rows = [describe_one(**s) for s in specs]
    table = pd.DataFrame(rows)

    core_cols = [
        "tag", "status", "n_rows", "n_cols", "n_unique_patients", "mean_rows_per_patient",
        "target_col", "label_prevalence", "n_label_positive", "n_label_negative",
        "age_mean", "age_sd", "overall_missing_rate",
    ]
    core_cols = [c for c in core_cols if c in table.columns]
    other_cols = [c for c in table.columns if c not in core_cols]
    table = table[core_cols + other_cols]

    table.to_csv(outdir / "table1_combined.csv", index=False)

    md = ["# Table 1 -- combined baseline characteristics\n"]
    core = table[core_cols].copy()
    for c in core.columns:
        if core[c].dtype == float:
            core[c] = core[c].round(4)
    md.append(to_markdown_table(core))
    md.append("\n\n## Full detail (including cohort-flow counts, if available)\n")
    full = table.copy()
    for c in full.columns:
        if full[c].dtype == float:
            full[c] = full[c].round(4)
    md.append(to_markdown_table(full))
    (outdir / "table1.md").write_text("\n".join(md), encoding="utf-8")

    print("\n[Done] Table 1 written to:", outdir)
    print(" - table1_combined.csv")
    print(" - table1.md")
    print("\n" + to_markdown_table(core))


if __name__ == "__main__":
    main()
