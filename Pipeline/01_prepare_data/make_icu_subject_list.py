#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_icu_subject_list.py
========================
Writes the list of MIMIC-IV Full patients (subject_id) who have at least one ICU
stay, read from the official release table icu/icustays.csv.gz.

WHY: every one of the 100 official MIMIC-IV Demo patients has an ICU
stay, against ~17% of Full patients. A null distribution that is meant to describe
"samples drawn the way the Demo was drawn" must therefore sample from these
patients only. compare_trustworthiness.py --null_pool icu --icu_subjects_path
<this file> does that.

The list is used ONLY to choose which Full patients may be sampled into a null
draw (and to define the Full-ICU reference population). It is never a model
feature: compare_trustworthiness.split_X_y builds the feature matrix from the
analytic CSV alone, which has no ICU column.

Usage:
  python make_icu_subject_list.py \
      --icustays "<...>/MIMIC_Dataset/MIMIC-IV-Full-2.2/icu/icustays.csv.gz" \
      --out      "<...>/Results/checks/full_icu_subject_ids.csv"

Outputs: the CSV (one column `subject_id`, sorted, unique) and <out>.json with the
source file's SHA-256, its row count and the number of patients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

# SHA-256 of icu/icustays.csv.gz in the official MIMIC-IV v2.2 release (its SHA256SUMS.txt).
OFFICIAL_V22_ICUSTAYS_SHA256 = "b7d37536d4ce6f68feb3c5a89068bec34c5f03a573162df02bd57aec53b87c7d"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icustays", required=True, help="Full release icu/icustays.csv.gz")
    ap.add_argument("--out", required=True, help="Output CSV (column subject_id)")
    ap.add_argument("--allow_unofficial", action="store_true",
                    help="Accept an icustays file whose SHA-256 differs from the official v2.2 file.")
    a = ap.parse_args()

    src = Path(a.icustays)
    sha = sha256_file(src)
    if sha != OFFICIAL_V22_ICUSTAYS_SHA256 and not a.allow_unofficial:
        raise SystemExit(f"[make_icu_subject_list] STOP: {src} SHA-256 {sha} is not the official MIMIC-IV v2.2 "
                         f"icustays.csv.gz ({OFFICIAL_V22_ICUSTAYS_SHA256}).")
    st = pd.read_csv(src, usecols=["subject_id"])
    ids = pd.to_numeric(st["subject_id"], errors="raise").astype("int64")
    out_ids = pd.DataFrame({"subject_id": sorted(set(ids.tolist()))})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_ids.to_csv(out, index=False)
    src_label = ("MIMIC-IV-Full-2.2/icu/icustays.csv.gz (official v2.2 file, SHA-256 verified)"
                 if sha == OFFICIAL_V22_ICUSTAYS_SHA256 else str(src))
    meta = {"source": src_label, "source_sha256": sha, "source_rows": int(len(st)),
            "n_subjects": int(len(out_ids)), "output_sha256": sha256_file(out)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[make_icu_subject_list] {meta['n_subjects']:,} ICU patients from {meta['source_rows']:,} ICU stays -> {out}")


if __name__ == "__main__":
    main()
