#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_ed.py
=============
Builds the Emergency-Department-module analytic dataset from an official
MIMIC-IV-ED release folder (e.g. MIMIC-IV-ED-Demo-2.2/ or
MIMIC-IV-ED-Full-2.2/, each with an `ed/` subfolder).

WHY THIS TARGET, NOT "in-hospital mortality" AGAIN
---------------------------------------------------
The Hospital module (prepare_hosp.py) predicts in-hospital mortality, because
that label (hospital_expire_flag) is defined for every hospital admission.

MIMIC-IV-ED is a different kind of table: it tracks *ED stays*, most of which
never become a hospital admission at all (edstays.hadm_id is only populated
for the stays that led to admission). Two options were considered:

  (a) Restrict to the subset of ED stays with hadm_id populated, link forward
      to hospital_expire_flag, and predict in-hospital mortality from ED-time
      features. This keeps the same label as the Hospital module, but throws
      away most of the ED table (only admitted stays), so it no longer
      exercises what MIMIC-IV-ED was built for, and the reference paper
      (Johnson et al., MIMIC-IV-ED) explicitly does not treat mortality as
      the natural ED-level task.

  (b) Predict ED disposition -- whether the stay resulted in hospital
      admission (hadm_id not null) versus discharge home -- using only
      information available at the point of the disposition decision
      (triage vitals/acuity, vitals recorded during the ED stay, home
      medication reconciliation, ED-administered medications, and basic
      demographics). This is the standard, well-powered task the ED table
      supports (it is available for 100% of stays, not a small linked
      subset), and it mirrors what MIMIC-IV-ED is actually used for in the
      literature (predicting admission at ED triage/disposition time).

This script implements (b): label_ed_admit = 1 if edstays.hadm_id is
populated, else 0.

ANTI-LEAKAGE NOTES (definitional and temporal leakage)
-------------------------------------------------------------------------------
1) The `diagnosis` table is deliberately EXCLUDED from the feature set. ED
   diagnoses are ICD codes assigned by trained coders *after* hospital
   discharge for billing purposes -- they are not available at ED disposition
   time, so using them as a predictor of the disposition decision would be a
   textbook case of definitional/temporal leakage (the "diagnosis" already
   encodes whether/how the patient was treated as an inpatient).

2) LANDMARK DESIGN (replaces the earlier fixed 4-hour window). A fixed window
   measured from ED arrival still leaked post-decision information: on
   MIMIC-IV-ED Demo, 45% of admitted stays already had a hospital admittime
   within intime+4h, so vitals/meds charted while the patient was boarding
   (already admitted in practice) were used to "predict" the admission. This
   script now uses a landmark time T = intime + --landmark_hours (default
   1.0h): (i) vitalsign/pyxis/medrecon events are restricted to charttime <=
   T, and (ii) stays that had ALREADY LEFT the ED (outtime <= T) or had
   ALREADY BEEN ADMITTED (hospital admittime <= T) are EXCLUDED, because at
   the landmark there is nothing left to predict for them. Every included
   stay therefore has the same observation window, so window length cannot
   act as a proxy for the label (truncating each stay at its own admittime
   would reintroduce label-dependent leakage through the count features).
   Rows with an unparseable/missing charttime are DROPPED rather than kept,
   since their eligibility for the window can't be verified. Exclusion counts
   are written to the .cohort_flow.json.

3) PLAUSIBILITY CLEANING. Raw MIMIC-IV-ED vitals contain data-entry/unit
   errors (on Full: triage dbp up to 661,672; o2sat up to 9,322; temperatures
   of 986 and hundreds recorded in Celsius; pain values of 13-180 on a 0-10
   scale). Before any aggregation, temperatures in the Celsius range
   (25-45) are converted to Fahrenheit, and every value outside the
   physiologic ranges in PLAUSIBLE_RANGES is set to missing (not clipped).
   Counts of values removed are logged and written to the .cohort_flow.json.

LABEL DEFINITION (precise): label_ed_admit = 1 when
edstays.hadm_id is populated, i.e. the ED stay resulted in a hospital
encounter -- an inpatient admission OR placement in hospital observation
(e.g. admission_type "EU OBSERVATION"; on Demo, 22 of 172 positive stays have
an edstays.disposition other than ADMITTED, mostly HOME after observation).
This is the conventional MIMIC-IV-ED "hospitalization" definition.

Feature set (1 row = 1 ED stay_id):
  - triage: temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity
    (triage is, by definition, recorded at ED arrival -- no window needed)
  - vitalsign (charted between arrival and the landmark): count of readings
    (0 if none), mean/min/max of temperature, heartrate, resprate, o2sat,
    sbp, dbp
  - medrecon (charted by the landmark): count of distinct home medications
    reconciled (distinct `name`; n_home_meds)
  - pyxis (charted by the landmark): count of distinct ED-dispensed
    medications (distinct `med_rn` -- MIMIC-IV-ED repeats a dispense row once
    per GSN code; n_ed_meds)
  - demographics (optional, --hosp_dir): gender, anchor_age, joined on
    subject_id from the matching Hospital-module patients table

subject_id is KEPT in the output (previously dropped) so downstream scripts
can use it for group-aware cross-validation (--group_col subject_id in
run_rsce.py / run_ppv.py), preventing the same patient's repeat ED visits
from being split across train and test. Drop it from the FEATURE set via
--drop_cols subject_id stay_id when running run_rsce.py / run_ppv.py -- it
is an identifier, not a predictor.

Usage:
  python prepare_ed.py \
      --ed_dir "/path/to/MIMIC-IV-ED-Demo-2.2" \
      --hosp_dir "/path/to/MIMIC-IV-Demo-2.2" \
      --out demo_ed_analytic_dataset_admission.csv

  python prepare_ed.py \
      --ed_dir "/path/to/MIMIC-IV-ED-Full-2.2" \
      --hosp_dir "/path/to/MIMIC-IV-Full-2.2" \
      --out full_ed_analytic_dataset_admission.csv \
      --chunksize 2000000 --landmark_hours 1.0

--hosp_dir is REQUIRED (hospital admittime is needed for the landmark
exclusion, and gender/anchor_age come from hosp/patients).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRIAGE_NUMERIC_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
VITALSIGN_NUMERIC_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

DEFAULT_LANDMARK_HOURS = 1.0

# Physiologic plausibility ranges applied to raw triage/vitalsign values
# BEFORE aggregation; values outside -> NaN. Temperature is in Fahrenheit
# (after converting Celsius-range entries, see clean_vitals()).
PLAUSIBLE_RANGES = {
    "temperature": (86.0, 113.0),   # 30-45 C
    "heartrate": (20.0, 300.0),
    "resprate": (2.0, 80.0),
    "o2sat": (40.0, 100.0),
    "sbp": (40.0, 300.0),
    "dbp": (10.0, 200.0),
    "pain": (0.0, 10.0),
    "acuity": (1.0, 5.0),
}
CELSIUS_RANGE = (25.0, 45.0)


def clean_vitals(df: pd.DataFrame, cols, counts: dict, prefix: str) -> pd.DataFrame:
    """Convert Celsius-range temperatures to F, then NaN-out implausible values (in place on a copy)."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if c == "temperature":
            is_c = v.between(*CELSIUS_RANGE)
            counts[f"{prefix}{c}_celsius_converted"] = counts.get(f"{prefix}{c}_celsius_converted", 0) + int(is_c.sum())
            v = v.where(~is_c, v * 9.0 / 5.0 + 32.0)
        lo, hi = PLAUSIBLE_RANGES[c]
        bad = v.notna() & ~v.between(lo, hi)
        counts[f"{prefix}{c}_implausible_set_nan"] = counts.get(f"{prefix}{c}_implausible_set_nan", 0) + int(bad.sum())
        df[c] = v.where(~bad)
    return df


def log(msg: str) -> None:
    print(f"[prepare_ed] {msg}")


def _read_table(ed_dir: Path, name: str, usecols=None, dtype=None) -> pd.DataFrame:
    path = ed_dir / f"{name}.csv.gz"
    if not path.exists():
        path = ed_dir / f"{name}.csv"
    log(f"Loading {name}: {path}")
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)


def load_edstays_and_label(ed_dir: Path) -> pd.DataFrame:
    edstays = _read_table(ed_dir, "edstays")
    for col in ["subject_id", "hadm_id", "stay_id"]:
        if col in edstays.columns:
            edstays[col] = pd.to_numeric(edstays[col], errors="coerce").astype("Int64")

    edstays["label_ed_admit"] = edstays["hadm_id"].notna().astype(int)
    for tcol in ("intime", "outtime"):
        if tcol not in edstays.columns:
            raise ValueError(f"edstays has no '{tcol}' column; the landmark design needs it.")
        edstays[tcol] = pd.to_datetime(edstays[tcol], errors="coerce")

    log(f"edstays shape: {edstays.shape}")
    log("Label value counts (label_ed_admit):\n" + str(edstays["label_ed_admit"].value_counts(dropna=False)))
    return edstays[["subject_id", "stay_id", "hadm_id", "intime", "outtime", "label_ed_admit"]].copy()


def load_admittime(hosp_dir: Path) -> pd.DataFrame:
    path = hosp_dir / "hosp" / "admissions.csv.gz"
    if not path.exists():
        path = hosp_dir / "hosp" / "admissions.csv"
    if not path.exists():
        raise FileNotFoundError(f"hosp/admissions.csv(.gz) not found under {hosp_dir}; needed for the landmark exclusion.")
    adm = pd.read_csv(path, usecols=["hadm_id", "admittime"])
    adm["hadm_id"] = pd.to_numeric(adm["hadm_id"], errors="coerce").astype("Int64")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    return adm.drop_duplicates(subset=["hadm_id"])


def apply_landmark_cohort(base: pd.DataFrame, adm: pd.DataFrame, landmark_hours: float, flow: dict) -> pd.DataFrame:
    """Exclude stays that, at intime + landmark_hours, had already left the ED or already been admitted."""
    b = base.merge(adm, on="hadm_id", how="left")
    T = b["intime"] + pd.to_timedelta(float(landmark_hours), unit="h")
    no_intime = b["intime"].isna()
    left_ed = b["outtime"].notna() & (b["outtime"] <= T)
    admitted = b["admittime"].notna() & (b["admittime"] <= T)
    excl = no_intime | left_ed | admitted
    y = b["label_ed_admit"] == 1
    flow["landmark_hours"] = float(landmark_hours)
    flow["n_excluded_missing_intime"] = int(no_intime.sum())
    flow["n_excluded_left_ed_before_landmark"] = int((left_ed & ~no_intime).sum())
    flow["n_excluded_admitted_before_landmark"] = int((admitted & ~left_ed & ~no_intime).sum())
    flow["n_excluded_total"] = int(excl.sum())
    flow["n_excluded_label1"] = int((excl & y).sum())
    flow["n_excluded_label0"] = int((excl & ~y).sum())
    flow["pct_label1_stays_kept"] = float(1 - (excl & y).sum() / max(1, y.sum()))
    flow["pct_label0_stays_kept"] = float(1 - (excl & ~y).sum() / max(1, (~y).sum()))
    kept = b.loc[~excl].drop(columns=["admittime"])
    log(f"Landmark {landmark_hours}h: kept {len(kept):,}/{len(b):,} stays "
        f"(excluded {flow['n_excluded_left_ed_before_landmark']:,} left ED, "
        f"{flow['n_excluded_admitted_before_landmark']:,} already admitted, "
        f"{flow['n_excluded_missing_intime']:,} missing intime).")
    return kept


def apply_time_window(
    df: pd.DataFrame,
    cutoff: pd.DataFrame,
    window_hours: float,
    time_col: str = "charttime",
) -> pd.DataFrame:
    """
    Keep only rows with `time_col` <= (edstays.intime + window_hours). `cutoff`
    must have columns [stay_id, intime]. Rows whose stay_id has no intime, or
    whose time_col can't be parsed, are dropped (conservative: can't verify
    they're pre-cutoff, so they're excluded rather than assumed safe).
    (window_hours = the landmark; must be > 0.) Stays not present in `cutoff`
    (i.e. excluded by the landmark cohort rule) are dropped by the inner merge.
    """
    if window_hours is None or window_hours <= 0:
        raise ValueError("landmark/window hours must be > 0")
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found; cannot apply the landmark restriction to this table.")

    n_before = len(df)
    d = df.merge(cutoff, on="stay_id", how="inner")
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    cutoff_time = d["intime"] + pd.to_timedelta(float(window_hours), unit="h")
    keep = (d[time_col] <= cutoff_time) & d[time_col].notna() & d["intime"].notna()
    d = d.loc[keep].drop(columns=["intime"])
    n_after = len(d)
    log(f"  time-window filter (<= intime + {window_hours}h) on '{time_col}': "
        f"{n_before:,} -> {n_after:,} rows kept ({n_before - n_after:,} dropped).")
    return d


def load_triage_features(ed_dir: Path, clean_counts: dict) -> pd.DataFrame:
    triage = _read_table(ed_dir, "triage")
    if "stay_id" in triage.columns:
        triage["stay_id"] = pd.to_numeric(triage["stay_id"], errors="coerce").astype("Int64")
    keep = ["stay_id"] + [c for c in TRIAGE_NUMERIC_COLS if c in triage.columns]
    triage = triage[keep].copy()
    # `pain` is a self-reported 0-10 scale but is stored as free text in MIMIC-IV-ED,
    # so a small number of rows contain non-numeric noise (e.g. "Critical", "o" for
    # "0"). Coerce to numeric so it is used as the ordinal scale it actually is,
    # rather than becoming a high-cardinality text column; unparseable entries -> NaN
    # (imputed downstream like any other missing lab/vital value).
    if "pain" in triage.columns:
        n_before = triage["pain"].notna().sum()
        triage["pain"] = pd.to_numeric(triage["pain"], errors="coerce")
        n_after = triage["pain"].notna().sum()
        if n_before != n_after:
            log(f"triage.pain: coerced {n_before - n_after} non-numeric entries to NaN ({n_after}/{len(triage)} numeric).")
    triage = clean_vitals(triage, TRIAGE_NUMERIC_COLS, clean_counts, prefix="triage_")
    triage = triage.rename(columns={c: f"triage_{c}" for c in TRIAGE_NUMERIC_COLS if c in triage.columns})
    triage = triage.drop_duplicates(subset=["stay_id"])
    log(f"triage feature shape: {triage.shape}")
    return triage


def load_vitalsign_features(
    ed_dir: Path, chunksize: Optional[int], cutoff: pd.DataFrame, window_hours: float, clean_counts: dict
) -> pd.DataFrame:
    path = ed_dir / "vitalsign.csv.gz"
    if not path.exists():
        path = ed_dir / "vitalsign.csv"
    if not path.exists():
        log("WARNING: vitalsign table not found; skipping vitalsign features.")
        return pd.DataFrame(columns=["stay_id"])

    usecols = ["stay_id", "charttime"] + [c for c in VITALSIGN_NUMERIC_COLS]
    log(f"Streaming vitalsign: {path} (chunksize={chunksize})")

    agg_parts = []
    n_before_total, n_after_total = 0, 0
    reader = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False, chunksize=chunksize)
    iterator = reader if chunksize else [reader]
    for chunk in iterator:
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        n_before_total += len(chunk)
        chunk = apply_time_window(chunk, cutoff, window_hours, time_col="charttime")
        chunk = clean_vitals(chunk, VITALSIGN_NUMERIC_COLS, clean_counts, prefix="vital_")
        n_after_total += len(chunk)
        agg_parts.append(chunk)
    vit = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame(columns=usecols)
    log(f"vitalsign total after landmark filter: {n_before_total:,} -> {n_after_total:,} rows.")

    if vit.empty:
        return pd.DataFrame(columns=["stay_id"])

    present_cols = [c for c in VITALSIGN_NUMERIC_COLS if c in vit.columns]
    agg_dict = {c: ["mean", "min", "max"] for c in present_cols}
    grouped = vit.groupby("stay_id").agg(agg_dict)
    grouped.columns = [f"vital_{col}_{stat}" for col, stat in grouped.columns]
    grouped["vital_n_readings"] = vit.groupby("stay_id").size()
    grouped = grouped.reset_index()
    log(f"vitalsign feature shape: {grouped.shape}")
    return grouped


def load_medrecon_counts(ed_dir: Path, cutoff: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    medrecon = _read_table(ed_dir, "medrecon", usecols=lambda c: c in ("stay_id", "name", "charttime"))
    medrecon["stay_id"] = pd.to_numeric(medrecon["stay_id"], errors="coerce").astype("Int64")
    medrecon = apply_time_window(medrecon, cutoff, window_hours, time_col="charttime")
    # one row per (medication x ETC code) in MIMIC-IV-ED -> count DISTINCT medication names
    counts = medrecon.groupby("stay_id")["name"].nunique().reset_index(name="n_home_meds")
    log(f"medrecon counts shape: {counts.shape}")
    return counts


def load_pyxis_counts(ed_dir: Path, cutoff: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    pyxis = _read_table(ed_dir, "pyxis", usecols=lambda c: c in ("stay_id", "med_rn", "name", "charttime"))
    pyxis["stay_id"] = pd.to_numeric(pyxis["stay_id"], errors="coerce").astype("Int64")
    pyxis = apply_time_window(pyxis, cutoff, window_hours, time_col="charttime")
    # one row per (dispensed medication x GSN code) -> count DISTINCT dispensed medications (med_rn)
    key = "med_rn" if "med_rn" in pyxis.columns else "name"
    counts = pyxis.groupby("stay_id")[key].nunique().reset_index(name="n_ed_meds")
    log(f"pyxis counts shape: {counts.shape}")
    return counts


def load_demographics(hosp_dir: Path) -> Optional[pd.DataFrame]:
    patients_path = hosp_dir / "hosp" / "patients.csv.gz"
    if not patients_path.exists():
        patients_path = hosp_dir / "hosp" / "patients.csv"
    if not patients_path.exists():
        log(f"WARNING: could not find hosp/patients.csv(.gz) under {hosp_dir}; skipping demographics.")
        return None
    patients = pd.read_csv(patients_path, usecols=lambda c: c in ("subject_id", "gender", "anchor_age"))
    patients["subject_id"] = pd.to_numeric(patients["subject_id"], errors="coerce").astype("Int64")
    log(f"demographics shape: {patients.shape}")
    return patients


def build_analytic_dataset(
    ed_dir: Path, hosp_dir: Path, chunksize: Optional[int], landmark_hours: float
) -> tuple[pd.DataFrame, dict]:
    flow: dict = {}
    clean_counts: dict = {}

    base_all = load_edstays_and_label(ed_dir)
    flow["n_ed_stays_raw"] = int(len(base_all))
    flow["n_label_ed_admit_positive_raw"] = int(base_all["label_ed_admit"].sum())

    base = apply_landmark_cohort(base_all, load_admittime(hosp_dir), landmark_hours, flow)
    flow["n_ed_stays_total"] = int(len(base))
    flow["n_label_ed_admit_positive"] = int(base["label_ed_admit"].sum())
    flow["n_label_ed_admit_negative"] = int((base["label_ed_admit"] == 0).sum())
    flow["n_unique_subjects"] = int(base["subject_id"].nunique())

    cutoff = base[["stay_id", "intime"]].drop_duplicates(subset=["stay_id"])

    triage = load_triage_features(ed_dir, clean_counts)
    vital = load_vitalsign_features(ed_dir, chunksize, cutoff, landmark_hours, clean_counts)
    medrecon = load_medrecon_counts(ed_dir, cutoff, landmark_hours)
    pyxis = load_pyxis_counts(ed_dir, cutoff, landmark_hours)

    df = base.drop(columns=["intime", "outtime", "hadm_id"]).merge(triage, on="stay_id", how="left")
    df = df.merge(vital, on="stay_id", how="left")
    df = df.merge(medrecon, on="stay_id", how="left")
    df = df.merge(pyxis, on="stay_id", how="left")

    df["n_home_meds"] = df["n_home_meds"].fillna(0)
    df["n_ed_meds"] = df["n_ed_meds"].fillna(0)
    if "vital_n_readings" in df.columns:
        df["vital_n_readings"] = df["vital_n_readings"].fillna(0)  # no readings in the window = 0, not missing

    demo = load_demographics(hosp_dir)
    if demo is not None:
        df = df.merge(demo, on="subject_id", how="left")

    flow["plausibility_cleaning"] = clean_counts
    log(f"Plausibility cleaning: {clean_counts}")

    # subject_id is intentionally KEPT (see module docstring) for group-aware CV.
    flow["n_rows_final"] = int(len(df))
    flow["n_cols_final"] = int(df.shape[1])
    log(f"Final ED analytic dataset shape: {df.shape}")
    missing_report = df.isna().mean().sort_values(ascending=False)
    log("Missing rate (top 15 columns):\n" + str(missing_report.head(15)))
    flow["missing_rate_top15"] = {k: float(v) for k, v in missing_report.head(15).items()}
    return df, flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the ED-module admission-disposition analytic dataset from an official MIMIC-IV-ED release folder.")
    p.add_argument("--ed_dir", type=str, required=True,
                    help="Path to a MIMIC-IV-ED release root (e.g. .../MIMIC-IV-ED-Demo-2.2 or .../MIMIC-IV-ED-Full-2.2). Must contain an 'ed' subfolder.")
    p.add_argument("--hosp_dir", type=str, required=True,
                    help="Path to the matching Hospital-module release root (e.g. .../MIMIC-IV-Demo-2.2): hosp/admissions "
                         "(admittime, for the landmark exclusion) and hosp/patients (gender, anchor_age).")
    p.add_argument("--out", type=str, required=True, help="Output analytic CSV path.")
    p.add_argument("--chunksize", type=int, default=None,
                    help="If set, stream vitalsign.csv.gz in chunks of this many rows. Recommended for the Full ED dataset (e.g. 2000000); vitalsign.csv.gz alone is ~25GB for Full.")
    p.add_argument("--landmark_hours", type=float, default=DEFAULT_LANDMARK_HOURS,
                    help=f"Landmark time after ED arrival (default {DEFAULT_LANDMARK_HOURS}h). Features use only events "
                         f"charted by intime + landmark; stays that left the ED or were admitted by then are excluded.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ed_dir = Path(args.ed_dir).expanduser().resolve()
    ed_sub = ed_dir / "ed"
    if not ed_sub.exists():
        raise FileNotFoundError(f"No 'ed' subfolder found under {ed_dir}")

    hosp_dir = Path(args.hosp_dir).expanduser().resolve()
    if args.landmark_hours <= 0:
        raise ValueError("--landmark_hours must be > 0")

    df, flow = build_analytic_dataset(ed_sub, hosp_dir, args.chunksize, args.landmark_hours)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log(f"Saved analytic dataset to: {out_path.resolve()}")

    flow_path = out_path.with_suffix("").with_name(out_path.stem + ".cohort_flow.json")
    flow_path.write_text(json.dumps(flow, indent=2), encoding="utf-8")
    log(f"Saved cohort-flow summary to: {flow_path.resolve()}")


if __name__ == "__main__":
    main()
