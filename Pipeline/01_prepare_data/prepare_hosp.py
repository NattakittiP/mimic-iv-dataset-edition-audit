#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_hosp.py
================
Builds the Hospital-module analytic dataset (in-hospital mortality) directly
from an official MIMIC-IV release folder (the PhysioNet .csv.gz layout, e.g.
MIMIC-IV-Demo-2.2/ or MIMIC-IV-Full-2.2/, each with a `hosp/` subfolder).

Compared with a workflow that expects three flat, pre-extracted CSVs
(patients.csv, admissions.csv, labevents.csv), this script:

  1) Reads directly from the official hosp/*.csv.gz files (no manual export
     step needed).
  2) Streams labevents.csv.gz in chunks and filters to the admitted-adult
     cohort as it goes, instead of loading the entire file into memory. This
     matters a lot for the Full dataset, where labevents.csv.gz alone is
     ~1.9 GB compressed (tens of GB decompressed).
  3) Is fully CLI-driven, so the same script runs unmodified against Demo or
     Full -- only --mimic_dir (and --out) change.

Cohort / label / features:
  - cohort = all admissions for adult patients (anchor_age >= 18)
  - label_mortality = hospital_expire_flag (0/1)
  - features = top-N most frequent lab itemids (median value per admission,
    over labs charted up to the landmark), pivoted wide, plus demographic /
    admission columns from patients/admissions.

LANDMARK DESIGN (default --landmark_hours 24).
Taking the median of every lab charted during the WHOLE admission would use
labs drawn in the last hours before death -- for an
in-hospital-mortality label that is temporal leakage (the model partly
"predicts" death from the physiology of dying). Now the prediction time is
T = admittime + --landmark_hours:
  (i)  only labs with charttime <= T are used (labs linked to the admission
       but drawn before admittime, e.g. in the ED, are available at T and kept);
  (ii) admissions that had already ENDED by T (dischtime <= T or
       deathtime <= T) are EXCLUDED -- at T there is nothing left to predict
       for them -- and so are admissions with a missing/unparseable
       admittime. Every included admission therefore has the same 24-hour
       observation window. Exclusion counts (by label) go to the
       .cohort_flow.json. --landmark_hours 0 gives the whole-admission
       design (leaks for mortality; not used for the reported results).
The label is unchanged (death at any time during the admission, after T).

Usage:
  python prepare_hosp.py \
      --mimic_dir "/path/to/MIMIC-IV-Demo-2.2" \
      --out demo_analytic_dataset_mortality_all_admissions.csv

  python prepare_hosp.py \
      --mimic_dir "/path/to/MIMIC-IV-Full-2.2" \
      --out full_analytic_dataset_mortality_all_admissions.csv \
      --chunksize 2000000

  # Demo, with EXACTLY the Full dataset's lab itemids:
  python prepare_hosp.py --mimic_dir "/path/to/MIMIC-IV-Demo-2.2" \
      --out demo_analytic_dataset_mortality_all_admissions.csv \
      --lab_itemids_from full_analytic_dataset_mortality_all_admissions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

TOP_N_LABS_DEFAULT = 30
DEFAULT_LANDMARK_HOURS = 24.0

CANDIDATE_DEMO_COLS = [
    "subject_id",
    "gender",
    "anchor_age",
    "race",
    "marital_status",
    "insurance",
    "admission_type",
    "admission_location",
    "discharge_location",
    "anchor_year",
    "anchor_year_group",
]

LABEVENTS_USECOLS = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
LABEVENTS_DTYPES = {
    "subject_id": "Int64",
    "hadm_id": "Int64",
    "itemid": "Int64",
    "charttime": "string",
    "valuenum": "float64",
}


def log(msg: str) -> None:
    print(f"[prepare_hosp] {msg}")


def load_patients_admissions(hosp_dir: Path) -> pd.DataFrame:
    patients_path = hosp_dir / "patients.csv.gz"
    admissions_path = hosp_dir / "admissions.csv.gz"
    if not patients_path.exists():
        # fall back to uncompressed, in case the user already extracted it
        patients_path = hosp_dir / "patients.csv"
    if not admissions_path.exists():
        admissions_path = hosp_dir / "admissions.csv"

    log(f"Loading patients: {patients_path}")
    patients = pd.read_csv(patients_path)
    log(f"Loading admissions: {admissions_path}")
    admissions = pd.read_csv(admissions_path)

    df_pa = admissions.merge(patients, on="subject_id", how="left")
    log(f"patients+admissions merged shape: {df_pa.shape}")
    return df_pa


def build_adult_cohort(df_pa: pd.DataFrame, flow: dict) -> pd.DataFrame:
    for col in ["subject_id", "hadm_id"]:
        if col in df_pa.columns:
            df_pa[col] = pd.to_numeric(df_pa[col], errors="coerce").astype("Int64")

    flow["n_admissions_raw"] = int(len(df_pa))
    flow["n_unique_subjects_raw"] = int(df_pa["subject_id"].nunique()) if "subject_id" in df_pa.columns else None

    if "anchor_age" in df_pa.columns:
        df_pa["anchor_age"] = pd.to_numeric(df_pa["anchor_age"], errors="coerce")
        cohort = df_pa[df_pa["anchor_age"] >= 18].copy()
        log(f"Cohort after age >= 18 filter: {cohort.shape}")
    else:
        log("WARNING: anchor_age not found; using all admissions (no age filter).")
        cohort = df_pa.copy()
    flow["n_admissions_after_age_filter"] = int(len(cohort))
    flow["n_excluded_by_age_filter"] = int(flow["n_admissions_raw"] - flow["n_admissions_after_age_filter"])

    if "hospital_expire_flag" in cohort.columns:
        cohort["label_mortality"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
    else:
        log("WARNING: hospital_expire_flag not found; label_mortality set to 0 for all rows.")
        cohort["label_mortality"] = 0

    flow["n_unique_subjects_cohort"] = int(cohort["subject_id"].nunique())
    flow["n_unique_hadm_cohort"] = int(cohort["hadm_id"].nunique())
    flow["n_label_positive"] = int(cohort["label_mortality"].sum())
    flow["n_label_negative"] = int((cohort["label_mortality"] == 0).sum())
    flow["label_prevalence"] = float(cohort["label_mortality"].mean())
    flow["mean_admissions_per_subject"] = float(flow["n_unique_hadm_cohort"] / flow["n_unique_subjects_cohort"]) if flow["n_unique_subjects_cohort"] else None

    log(f"Unique subject_id: {cohort['subject_id'].nunique()}, unique hadm_id: {cohort['hadm_id'].nunique()}")
    log("Label value counts:\n" + str(cohort["label_mortality"].value_counts(dropna=False)))
    return cohort


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


def apply_landmark_cohort(cohort: pd.DataFrame, landmark_hours: float, flow: dict) -> pd.DataFrame:
    """Exclude admissions that ended (discharge or death) by admittime + landmark_hours.

    Adds a 'landmark_time' column (used to restrict labs). landmark_hours <= 0
    disables the landmark (whole-admission labs).
    """
    flow["landmark_hours"] = float(landmark_hours)
    if landmark_hours <= 0:
        log("Landmark DISABLED (--landmark_hours 0): labs from the whole admission are used "
            "(temporal leakage for mortality; not used for the reported results).")
        cohort = cohort.copy()
        cohort["landmark_time"] = pd.NaT
        return cohort
    c = cohort.copy()
    admit = _to_dt(c["admittime"])
    disch = _to_dt(c["dischtime"]) if "dischtime" in c.columns else pd.Series(pd.NaT, index=c.index)
    death = _to_dt(c["deathtime"]) if "deathtime" in c.columns else pd.Series(pd.NaT, index=c.index)
    T = admit + pd.to_timedelta(landmark_hours, unit="h")
    miss = admit.isna()
    ended = (~miss) & ((disch.notna() & (disch <= T)) | (death.notna() & (death <= T)))
    excl = miss | ended
    y = c["label_mortality"]
    flow["n_excluded_missing_admittime"] = int(miss.sum())
    flow["n_excluded_ended_before_landmark"] = int(ended.sum())
    flow["n_excluded_ended_before_landmark_label1"] = int((ended & (y == 1)).sum())
    flow["n_excluded_ended_before_landmark_label0"] = int((ended & (y == 0)).sum())
    flow["pct_label1_admissions_kept"] = float(((~excl) & (y == 1)).sum() / max(int((y == 1).sum()), 1))
    flow["pct_label0_admissions_kept"] = float(((~excl) & (y == 0)).sum() / max(int((y == 0).sum()), 1))
    c["landmark_time"] = T
    out = c[~excl].copy()
    log(f"Landmark {landmark_hours:g}h: kept {len(out):,}/{len(c):,} admissions (excluded "
        f"{int(ended.sum()):,} discharged/died by the landmark [deaths {int((ended & (y == 1)).sum()):,}], "
        f"{int(miss.sum()):,} missing admittime).")
    flow["n_admissions_after_landmark"] = int(len(out))
    flow["n_label_positive_after_landmark"] = int(out["label_mortality"].sum())
    flow["n_label_negative_after_landmark"] = int((out["label_mortality"] == 0).sum())
    flow["label_prevalence_after_landmark"] = float(out["label_mortality"].mean()) if len(out) else None
    return out


def stream_top_lab_features(
    hosp_dir: Path,
    valid_hadm_ids: set,
    top_n_labs: int,
    chunksize: Optional[int],
    fixed_itemids: Optional[List[int]] = None,
    landmark_by_hadm: Optional[pd.Series] = None,
    flow: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Stream labevents.csv.gz, keep only rows for admissions in the cohort,
    then aggregate (median per hadm_id, itemid) and pivot to wide format
    using only the top-N most frequent itemids -- OR, if `fixed_itemids` is
    given, exactly that list of itemids (in that order), with an all-NaN
    column for any itemid that has no rows in this dataset.

    WHY fixed_itemids EXISTS: the top-N list computed separately on Demo and
    on Full is NOT the same (on MIMIC-IV v2.2, Demo's top-30 contains blood-gas
    itemids 50818/50820/50821 while Full's contains 50861/50878/50885), so a
    Demo-vs-Full comparison would silently compare two different feature
    sets. For the comparison, Demo must be built with Full's lab list:
    use --lab_itemids_from <the Full analytic CSV>.
    """
    labevents_path = hosp_dir / "labevents.csv.gz"
    if not labevents_path.exists():
        labevents_path = hosp_dir / "labevents.csv"
    log(f"Streaming labevents: {labevents_path} (chunksize={chunksize})")

    kept_chunks: List[pd.DataFrame] = []
    n_rows_seen = 0
    n_rows_kept = 0
    n_after_landmark_dropped = 0
    n_bad_charttime = 0

    reader = pd.read_csv(
        labevents_path,
        usecols=LABEVENTS_USECOLS,
        dtype=LABEVENTS_DTYPES,
        low_memory=False,
        chunksize=chunksize,
    )
    iterator = reader if chunksize else [reader]
    for chunk in iterator:
        n_rows_seen += len(chunk)
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk = chunk[chunk["hadm_id"].astype("Int64").isin(valid_hadm_ids)]
        if len(chunk) and landmark_by_hadm is not None:
            ct = pd.to_datetime(chunk["charttime"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
            T = chunk["hadm_id"].map(landmark_by_hadm)
            bad = ct.isna()
            keep = (~bad) & (ct <= T)
            n_bad_charttime += int(bad.sum())
            n_after_landmark_dropped += int(((~bad) & ~keep).sum())
            chunk = chunk[keep.to_numpy()]
        if len(chunk):
            kept_chunks.append(chunk[["hadm_id", "itemid", "valuenum"]])
            n_rows_kept += len(chunk)

    log(f"labevents rows seen={n_rows_seen:,}, kept for cohort={n_rows_kept:,}")
    if landmark_by_hadm is not None:
        log(f"  landmark filter: dropped {n_after_landmark_dropped:,} cohort lab rows charted after the landmark "
            f"and {n_bad_charttime:,} with missing/unparseable charttime.")
        if flow is not None:
            flow["lab_rows_dropped_after_landmark"] = int(n_after_landmark_dropped)
            flow["lab_rows_dropped_bad_charttime"] = int(n_bad_charttime)
            flow["lab_rows_kept"] = int(n_rows_kept)

    if not kept_chunks:
        log("WARNING: no lab rows matched the cohort; returning empty lab feature table.")
        return pd.DataFrame(columns=["hadm_id"])

    labs = pd.concat(kept_chunks, ignore_index=True)

    if fixed_itemids:
        top_itemids = [int(i) for i in fixed_itemids]
        present = set(int(i) for i in labs["itemid"].dropna().unique())
        missing = [i for i in top_itemids if i not in present]
        log(f"Using FIXED list of {len(top_itemids)} lab itemids as features: {top_itemids}")
        if missing:
            log(f"WARNING: {len(missing)} fixed itemid(s) have no rows in this dataset -> all-NaN columns: {missing}")
    else:
        lab_counts = labs["itemid"].value_counts()
        top_itemids = lab_counts.head(top_n_labs).index.tolist()
        log(f"Using TOP {top_n_labs} lab itemids as features: {top_itemids}")

    labs_top = labs[labs["itemid"].isin(top_itemids)]
    lab_agg = (
        labs_top.groupby(["hadm_id", "itemid"])["valuenum"]
        .median()
        .reset_index()
    )
    lab_wide = lab_agg.pivot(index="hadm_id", columns="itemid", values="valuenum")
    if fixed_itemids:
        # same columns, same (sorted) order as a top-N pivot would produce
        lab_wide = lab_wide.reindex(columns=sorted(int(i) for i in top_itemids))
    lab_wide.columns = [f"lab_{int(c)}" for c in lab_wide.columns]
    lab_wide = lab_wide.reset_index()
    log(f"lab_wide shape: {lab_wide.shape}")
    return lab_wide


def build_analytic_dataset(hosp_dir: Path, top_n_labs: int, chunksize: Optional[int],
                           fixed_itemids: Optional[List[int]] = None,
                           landmark_hours: float = DEFAULT_LANDMARK_HOURS) -> tuple[pd.DataFrame, dict]:
    flow: dict = {"top_n_labs": top_n_labs}
    if fixed_itemids:
        flow["lab_itemids_fixed"] = [int(i) for i in fixed_itemids]
    df_pa = load_patients_admissions(hosp_dir)
    cohort = build_adult_cohort(df_pa, flow)
    cohort = apply_landmark_cohort(cohort, landmark_hours, flow)

    valid_hadm_ids = set(cohort["hadm_id"].dropna().astype("Int64").tolist())
    landmark_by_hadm = None
    if landmark_hours > 0:
        landmark_by_hadm = cohort.drop_duplicates("hadm_id").set_index("hadm_id")["landmark_time"]
    lab_wide = stream_top_lab_features(hosp_dir, valid_hadm_ids, top_n_labs, chunksize, fixed_itemids,
                                       landmark_by_hadm=landmark_by_hadm, flow=flow)
    flow["n_hadm_with_any_lab_feature"] = int(lab_wide["hadm_id"].nunique()) if "hadm_id" in lab_wide.columns else 0

    demo_cols = [c for c in CANDIDATE_DEMO_COLS if c in cohort.columns]
    log(f"Demographic columns used: {demo_cols}")

    cohort_unique = (
        cohort.sort_values(by=["hadm_id"]).drop_duplicates(subset=["hadm_id"]).copy()
    )
    demo_cols_final = ["hadm_id"] + demo_cols + ["label_mortality"]
    df_demo = cohort_unique[demo_cols_final].copy()

    if "hadm_id" in lab_wide.columns:
        df_analytic = df_demo.merge(lab_wide, on="hadm_id", how="left")
    else:
        log("WARNING: lab_wide has no hadm_id column; analytic dataset will have no lab features.")
        df_analytic = df_demo.copy()

    flow["n_rows_final"] = int(len(df_analytic))
    flow["n_cols_final"] = int(df_analytic.shape[1])
    log(f"Final analytic dataset shape: {df_analytic.shape}")
    missing_report = df_analytic.isna().mean().sort_values(ascending=False)
    log("Missing rate (top 15 columns):\n" + str(missing_report.head(15)))
    flow["missing_rate_top15"] = {k: float(v) for k, v in missing_report.head(15).items()}
    return df_analytic, flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Hospital-module mortality analytic dataset from an official MIMIC-IV release folder.")
    p.add_argument("--mimic_dir", type=str, required=True,
                    help="Path to a MIMIC-IV release root (e.g. .../MIMIC-IV-Demo-2.2 or .../MIMIC-IV-Full-2.2). Must contain a 'hosp' subfolder.")
    p.add_argument("--out", type=str, required=True, help="Output analytic CSV path.")
    p.add_argument("--top_n_labs", type=int, default=TOP_N_LABS_DEFAULT, help="Number of most-frequent lab itemids to use as features.")
    p.add_argument("--chunksize", type=int, default=None,
                    help="If set, stream labevents.csv.gz in chunks of this many rows instead of loading it all at once. Recommended for the Full dataset (e.g. 2000000).")
    p.add_argument("--lab_itemids_from", type=str, default=None,
                    help="Path to an existing analytic CSV (normally the FULL one). Its lab_<itemid> columns are "
                         "used as the exact lab feature list instead of this dataset's own top-N. Use this when "
                         "building Demo so Demo and Full have identical feature sets.")
    p.add_argument("--landmark_hours", type=float, default=DEFAULT_LANDMARK_HOURS,
                    help="Prediction time = admittime + this many hours (default 24): only labs charted by then are "
                         "used and admissions discharged/died by then are excluded. 0 = whole-admission labs "
                         "(leaks for mortality; not used for the reported results).")
    return p.parse_args()


def itemids_from_csv(path: str) -> List[int]:
    cols = pd.read_csv(path, nrows=0).columns
    ids = [int(c[len("lab_"):]) for c in cols if c.startswith("lab_")]
    if not ids:
        raise ValueError(f"No lab_<itemid> columns found in {path}")
    return ids


def main() -> None:
    args = parse_args()
    mimic_dir = Path(args.mimic_dir).expanduser().resolve()
    hosp_dir = mimic_dir / "hosp"
    if not hosp_dir.exists():
        raise FileNotFoundError(f"No 'hosp' subfolder found under {mimic_dir}")

    fixed = itemids_from_csv(args.lab_itemids_from) if args.lab_itemids_from else None
    df_analytic, flow = build_analytic_dataset(hosp_dir, args.top_n_labs, args.chunksize, fixed,
                                               landmark_hours=args.landmark_hours)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_analytic.to_csv(out_path, index=False)
    log(f"Saved analytic dataset to: {out_path.resolve()}")

    flow_path = out_path.with_suffix("").with_name(out_path.stem + ".cohort_flow.json")
    flow_path.write_text(json.dumps(flow, indent=2), encoding="utf-8")
    log(f"Saved cohort-flow summary to: {flow_path.resolve()}")


if __name__ == "__main__":
    main()
