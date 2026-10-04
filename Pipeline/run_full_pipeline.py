#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
=====================
One-command orchestrator for the whole Pipeline/ folder: runs every stage in
the order documented in the repository README.md (prepare -> feasibility
check -> RSCE -> PPV -> Demo-vs-Full comparisons (incl. the ICU-patient null)
-> cross-domain synthesis -> Table 1 -> derived analyses -> figures ->
reproducibility snapshot), for BOTH the Hospital module and the ED module.

Written as a plain Python script (not a shell/bat script) so it runs
identically regardless of your terminal (PowerShell, cmd, bash, ...) -- just:

    python run_full_pipeline.py --list                 # show the plan, run nothing
    python run_full_pipeline.py --demo_only             # safe: Demo-scale only (fast)
    python run_full_pipeline.py                         # prepares Full data + runs
                                                          # --estimate_only feasibility
                                                          # checks, then STOPS before
                                                          # committing to the expensive
                                                          # Full-scale RSCE/PPV/compare
                                                          # runs (prints the estimate)
    python run_full_pipeline.py --run_full              # actually runs Full-scale
                                                          # RSCE/PPV/compare too (this
                                                          # can take hours; check the
                                                          # estimate first)

Safety rule: the expensive Full-scale RSCE/PPV/compare stages never run
unless you pass --run_full explicitly. Everything else (data prep for both
Demo and Full, the feasibility --estimate_only probes, the Demo-scale RSCE/
PPV/compare runs, Table 1, the Demo-subset check) always runs, since none of
those are slow.

Run this from inside the Pipeline/ folder (the one containing
01_prepare_data/, 02_rsce_benchmark/, etc.) with the environment from environment.yml
active. Assumes the sibling layout documented in the repository README.md:

  <project_root>/
  ├── MIMIC_Dataset/
  │   ├── MIMIC-IV-Demo-2.2/       MIMIC-IV-Full-2.2/
  │   └── MIMIC-IV-ED-Demo-2.2/    MIMIC-IV-ED-Full-2.2/
  ├── Results/                      <- created automatically
  └── Pipeline/                     <- this script lives here
      ├── run_full_pipeline.py
      ├── 01_prepare_data/ ...

Every command actually run, its exit code, and its wall-clock time are
appended to Results/run_full_pipeline.log so a long run can be audited (or
resumed with --stage) after the fact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PIPELINE_ROOT = Path(__file__).resolve().parent
PY = sys.executable

LOG_LINES: List[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


def flush_log(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run_full_pipeline.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    LOG_LINES.clear()


def banner(title: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {title}")
    log("=" * 78)


class StepFailed(RuntimeError):
    pass


def run(cmd: List[str], *, allow_missing_input: Optional[Path] = None, args_ns=None) -> bool:
    """
    Run one subprocess command. Returns True if it ran (whether it succeeded
    or -- with --continue_on_error -- failed), False if it was SKIPPED
    because a required input path doesn't exist (e.g. Full-scale MIMIC data
    not staged on this machine).
    """
    if allow_missing_input is not None and not allow_missing_input.exists():
        log(f"[SKIP] input not found, skipping this step: {allow_missing_input}")
        return False

    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log(f"\n[RUN] {printable}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    log(f"[{status}] in {dt:,.1f}s")

    if result.returncode != 0:
        if args_ns is not None and getattr(args_ns, "continue_on_error", False):
            log("[WARN] --continue_on_error set: continuing despite failure.")
        else:
            raise StepFailed(f"Command failed (exit {result.returncode}): {printable}")
    return True


def script(*parts: str) -> str:
    return str(PIPELINE_ROOT.joinpath(*parts))


def csv_prevalence(csv_path: Path, target_col: str) -> Optional[float]:
    """Read just the target column and compute its mean (prepare_hosp.py /
    prepare_ed.py both write label_* as already-coerced 0/1 ints, so a plain
    mean is the prevalence -- this mirrors what run_ppv.py would print as
    pi_ref when --pi_ref is omitted, computed directly instead of scraping
    subprocess stdout)."""
    try:
        import pandas as pd  # local import: only needed for this helper
        df = pd.read_csv(csv_path, usecols=[target_col])
        return float(df[target_col].astype(int).mean())
    except Exception as e:
        log(f"[WARN] could not compute pi_ref from {csv_path}: {e}")
        return None


def row_count(csv_path: Path) -> Optional[int]:
    try:
        import pandas as pd
        return int(len(pd.read_csv(csv_path, usecols=[0])))
    except Exception as e:
        log(f"[WARN] could not count rows in {csv_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def stage_prepare(a) -> None:
    banner("STAGE: prepare (01_prepare_data)")

    hosp_full_csv = a.out_root / "hosp_full" / "full_analytic_dataset_mortality_all_admissions.csv"

    # Full first: the Demo hosp dataset must use EXACTLY the Full dataset's
    # lab itemids (--lab_itemids_from), otherwise Demo's own "top-30 labs"
    # differ from Full's and the two are not comparable feature sets.
    if not a.demo_only:
        run([PY, script("01_prepare_data", "prepare_hosp.py"),
             "--mimic_dir", str(a.mimic_root / "MIMIC-IV-Full-2.2"),
             "--out", str(hosp_full_csv),
             "--chunksize", str(a.chunksize), "--landmark_hours", str(a.hosp_landmark_hours)],
            allow_missing_input=a.mimic_root / "MIMIC-IV-Full-2.2", args_ns=a)

        run([PY, script("01_prepare_data", "prepare_ed.py"),
             "--ed_dir", str(a.mimic_root / "MIMIC-IV-ED-Full-2.2"),
             "--hosp_dir", str(a.mimic_root / "MIMIC-IV-Full-2.2"),
             "--out", str(a.out_root / "ed_full" / "full_ed_analytic_dataset_admission.csv"),
             "--chunksize", str(a.chunksize), "--landmark_hours", str(a.landmark_hours)],
            allow_missing_input=a.mimic_root / "MIMIC-IV-ED-Full-2.2", args_ns=a)
    else:
        log("[INFO] --demo_only: skipping Full-scale data preparation.")

    demo_hosp_cmd = [PY, script("01_prepare_data", "prepare_hosp.py"),
                     "--mimic_dir", str(a.mimic_root / "MIMIC-IV-Demo-2.2"),
                     "--out", str(a.out_root / "hosp_demo" / "demo_analytic_dataset_mortality_all_admissions.csv"),
                     "--landmark_hours", str(a.hosp_landmark_hours)]
    # Fallback when the Full hosp CSV is absent (e.g. a Demo-only reproduction):
    # 01_prepare_data/full_lab_itemids.csv is a header-only file listing the 30
    # lab_<itemid> columns of the Full hosp dataset, in the same order, so the
    # Demo hosp dataset gets exactly the Full feature set. It holds no data rows.
    lab_list = PIPELINE_ROOT / "01_prepare_data" / "full_lab_itemids.csv"
    if hosp_full_csv.exists():
        demo_hosp_cmd += ["--lab_itemids_from", str(hosp_full_csv)]
    elif lab_list.exists():
        log(f"[INFO] Full hosp CSV not found: using the Full lab feature list in {lab_list}.")
        demo_hosp_cmd += ["--lab_itemids_from", str(lab_list)]
    else:
        log("[WARN] Full hosp CSV not found: the Demo hosp dataset will use Demo's OWN top labs, which "
            "differ from Full's. Fine for a --demo_only smoke test, NOT for a Demo-vs-Full comparison.")
    run(demo_hosp_cmd, args_ns=a)

    run([PY, script("01_prepare_data", "prepare_ed.py"),
         "--ed_dir", str(a.mimic_root / "MIMIC-IV-ED-Demo-2.2"),
         "--hosp_dir", str(a.mimic_root / "MIMIC-IV-Demo-2.2"),
         "--out", str(a.out_root / "ed_demo" / "demo_ed_analytic_dataset_admission.csv"),
         "--landmark_hours", str(a.landmark_hours)],
        args_ns=a)


def stage_checks(a) -> None:
    banner("STAGE: checks (Demo ⊂ Full subset verification)")
    if a.demo_only:
        log("[INFO] --demo_only: Demo-subset-of-Full check needs Full data; skipping.")
        return

    run([PY, script("01_prepare_data", "check_demo_subset_of_full.py"),
         "--demo_dir", str(a.mimic_root / "MIMIC-IV-Demo-2.2"),
         "--full_dir", str(a.mimic_root / "MIMIC-IV-Full-2.2"),
         "--module", "hosp", "--outdir", str(a.out_root / "checks")],
        allow_missing_input=a.mimic_root / "MIMIC-IV-Full-2.2", args_ns=a)

    run([PY, script("01_prepare_data", "check_demo_subset_of_full.py"),
         "--demo_dir", str(a.mimic_root / "MIMIC-IV-ED-Demo-2.2"),
         "--full_dir", str(a.mimic_root / "MIMIC-IV-ED-Full-2.2"),
         "--module", "ed", "--outdir", str(a.out_root / "checks")],
        allow_missing_input=a.mimic_root / "MIMIC-IV-ED-Full-2.2", args_ns=a)


def stage_estimate(a) -> None:
    banner("STAGE: estimate (feasibility probes for Full-scale RSCE/PPV)")
    if a.demo_only:
        log("[INFO] --demo_only: no Full-scale run to estimate; skipping.")
        return
    if a.run_full:
        # each probe runs 2 real folds on Full data (~2/15 of a whole job); pointless once committed
        log("[INFO] --run_full given: skipping the --estimate_only feasibility probes.")
        return

    hosp_full_csv = a.out_root / "hosp_full" / "full_analytic_dataset_mortality_all_admissions.csv"
    ed_full_csv = a.out_root / "ed_full" / "full_ed_analytic_dataset_admission.csv"

    extra = []
    if a.exclude_models:
        extra += ["--exclude_models"] + a.exclude_models
    if a.svc_max_train_n:
        extra += ["--svc_max_train_n", str(a.svc_max_train_n)]
    rsce_extra = extra + (["--no-compute_shap"] if a.no_shap else ["--shap_samples", str(a.shap_samples)])

    run([PY, script("02_rsce_benchmark", "run_rsce.py"),
         "--data", str(hosp_full_csv), "--target", "label_mortality",
         "--drop_cols", "hadm_id", "subject_id", "discharge_location", "anchor_year", "anchor_year_group",
         "--outdir", str(a.out_root / "hosp_full" / "rsce_estimate"),
         "--folds", str(a.folds), "--repeats", str(a.repeats), "--estimate_only"] + rsce_extra,
        allow_missing_input=hosp_full_csv, args_ns=a)

    run([PY, script("02_rsce_benchmark", "run_rsce.py"),
         "--data", str(ed_full_csv), "--target", "label_ed_admit",
         "--drop_cols", "stay_id", "subject_id",
         "--outdir", str(a.out_root / "ed_full" / "rsce_estimate"),
         "--folds", str(a.folds), "--repeats", str(a.repeats), "--estimate_only"] + rsce_extra,
        allow_missing_input=ed_full_csv, args_ns=a)

    run([PY, script("04_ppv", "run_ppv.py"),
         "--data", str(hosp_full_csv), "--target", "label_mortality",
         "--outdir", str(a.out_root / "hosp_full" / "ppv_estimate"),
         "--drop_cols"] + _ppv_drop_cols() +
        ["--folds", str(a.folds), "--repeats", str(a.repeats), "--estimate_only"] + extra,
        allow_missing_input=hosp_full_csv, args_ns=a)

    run([PY, script("04_ppv", "run_ppv.py"),
         "--data", str(ed_full_csv), "--target", "label_ed_admit",
         "--outdir", str(a.out_root / "ed_full" / "ppv_estimate"),
         "--drop_cols"] + _ppv_drop_cols() +
        ["--folds", str(a.folds), "--repeats", str(a.repeats), "--estimate_only"] + extra,
        allow_missing_input=ed_full_csv, args_ns=a)

    if not a.run_full:
        log("")
        log("[INFO] Feasibility estimates written under Results/*/rsce_estimate and */ppv_estimate.")
        log("[INFO] Review estimate_timing.csv in each, then re-run with --run_full to actually")
        log("[INFO] execute the Full-scale RSCE/PPV/compare stages (this can take hours).")
        log("[INFO] Tune with --exclude_models / --svc_max_train_n / --shap_samples / --no_shap")
        log("[INFO] if the estimate is too slow.")


def _rsce_drop_cols(domain: str) -> List[str]:
    if domain == "hosp":
        return ["hadm_id", "subject_id", "discharge_location", "anchor_year", "anchor_year_group"]
    return ["stay_id", "subject_id"]


def _ppv_drop_cols() -> List[str]:
    # Must include discharge_location (post-outcome label leakage for hosp
    # mortality) and match RSCE's hosp drop set; names absent from a given
    # dataset (e.g. hadm_id / discharge_location in ED data) are ignored by run_ppv.py.
    return ["hadm_id", "stay_id", "subject_id", "discharge_location", "anchor_year", "anchor_year_group"]


def stage_rsce(a) -> None:
    banner("STAGE: rsce (02_rsce_benchmark)")

    extra = []
    if a.exclude_models:
        extra += ["--exclude_models"] + a.exclude_models
    if a.svc_max_train_n:
        extra += ["--svc_max_train_n", str(a.svc_max_train_n)]
    if a.no_shap:
        extra += ["--no-compute_shap"]
    else:
        extra += ["--shap_samples", str(a.shap_samples)]

    domains = [
        ("hosp", "label_mortality", "demo_analytic_dataset_mortality_all_admissions.csv",
         "full_analytic_dataset_mortality_all_admissions.csv"),
        ("ed", "label_ed_admit", "demo_ed_analytic_dataset_admission.csv",
         "full_ed_analytic_dataset_admission.csv"),
    ]

    for domain, target, demo_name, full_name in domains:
        demo_csv = a.out_root / f"{domain}_demo" / demo_name
        run([PY, script("02_rsce_benchmark", "run_rsce.py"),
             "--data", str(demo_csv), "--target", target,
             "--drop_cols"] + _rsce_drop_cols(domain) +
            ["--outdir", str(a.out_root / f"{domain}_demo" / "rsce"),
             "--folds", str(a.folds), "--repeats", str(a.repeats)] + extra,
            allow_missing_input=demo_csv, args_ns=a)

        if a.demo_only or not a.run_full:
            continue

        full_csv = a.out_root / f"{domain}_full" / full_name
        run([PY, script("02_rsce_benchmark", "run_rsce.py"),
             "--data", str(full_csv), "--target", target,
             "--drop_cols"] + _rsce_drop_cols(domain) +
            ["--outdir", str(a.out_root / f"{domain}_full" / "rsce"),
             "--folds", str(a.folds), "--repeats", str(a.repeats)] + extra,
            allow_missing_input=full_csv, args_ns=a)


def stage_ppv(a) -> None:
    banner("STAGE: ppv (04_ppv)")

    extra = []
    if a.exclude_models:
        extra += ["--exclude_models"] + a.exclude_models
    if a.svc_max_train_n:
        extra += ["--svc_max_train_n", str(a.svc_max_train_n)]

    # run_ppv.py's --fast changes the MODEL ZOO itself (RF/ET 300 vs 1200
    # trees, GB 250 vs 700, MLP 800 vs 2500 iters). It is only acceptable for
    # a --demo_only plumbing smoke test. Whenever Demo is run as the
    # comparison partner of a Full run, Demo MUST use the same (non-fast)
    # models as Full, otherwise compare_ppv.py compares two different model
    # configurations. (Previous default turned --fast on for every Demo call,
    # including the comparison one -- fixed.) --fast/--no-fast still override.
    def fast_flag(is_demo: bool) -> List[str]:
        want_fast = a.fast if a.fast is not None else (is_demo and a.demo_only)
        return ["--fast"] if want_fast else []

    domains = [
        ("hosp", "label_mortality", "demo_analytic_dataset_mortality_all_admissions.csv",
         "full_analytic_dataset_mortality_all_admissions.csv"),
        ("ed", "label_ed_admit", "demo_ed_analytic_dataset_admission.csv",
         "full_ed_analytic_dataset_admission.csv"),
    ]

    for domain, target, demo_name, full_name in domains:
        demo_csv = a.out_root / f"{domain}_demo" / demo_name
        full_csv = a.out_root / f"{domain}_full" / full_name

        use_full_pi = (not a.demo_only) and a.run_full and full_csv.exists()
        if use_full_pi:
            pi_ref = csv_prevalence(full_csv, target)
            log(f"[INFO] {domain}: Demo PPV is standardized to Full's prevalence via --pi_ref_from "
                f"(~{pi_ref if pi_ref is not None else float('nan'):.8f}).")
            run([PY, script("04_ppv", "run_ppv.py"),
                 "--data", str(full_csv), "--target", target,
                 "--outdir", str(a.out_root / f"{domain}_full" / "ppv"),
                 "--drop_cols"] + _ppv_drop_cols() +
                ["--folds", str(a.folds), "--repeats", str(a.repeats)] + extra + fast_flag(False),
                allow_missing_input=full_csv, args_ns=a)

        demo_extra = list(extra)
        if use_full_pi:
            # computed by run_ppv.py from the Full CSV itself -> bit-identical to the Full run's pi_ref
            demo_extra += ["--pi_ref_from", str(full_csv)]
        run([PY, script("04_ppv", "run_ppv.py"),
             "--data", str(demo_csv), "--target", target,
             "--outdir", str(a.out_root / f"{domain}_demo" / "ppv"),
             "--drop_cols"] + _ppv_drop_cols() +
            ["--folds", str(a.folds), "--repeats", str(a.repeats)] + demo_extra + fast_flag(True),
            allow_missing_input=demo_csv, args_ns=a)


def stage_compare(a) -> None:
    banner("STAGE: compare (03_compare_demo_vs_full + 04_ppv/compare_ppv.py)")
    if a.demo_only or not a.run_full:
        log("[INFO] Demo-vs-Full comparisons need both sides at Full scale; "
            "skipping (pass --run_full, without --demo_only, to enable).")
        return

    domains = [
        ("hosp", "label_mortality", "demo_analytic_dataset_mortality_all_admissions.csv",
         "full_analytic_dataset_mortality_all_admissions.csv"),
        ("ed", "label_ed_admit", "demo_ed_analytic_dataset_admission.csv",
         "full_ed_analytic_dataset_admission.csv"),
    ]

    for domain, target, demo_name, full_name in domains:
        demo_rsce = a.out_root / f"{domain}_demo" / "rsce"
        full_rsce = a.out_root / f"{domain}_full" / "rsce"
        compare_dir = a.out_root / "compare" / domain
        base_dir = compare_dir / "_base"

        if not (demo_rsce / "rsce_scores.csv").exists() or not (full_rsce / "rsce_scores.csv").exists():
            log(f"[SKIP] {domain}: missing RSCE outputs for demo and/or full; skipping compare stage for {domain}.")
            continue

        run([PY, script("03_compare_demo_vs_full", "make_compare_base.py"),
             "--demo_dir", str(demo_rsce), "--full_dir", str(full_rsce),
             "--outdir", str(base_dir)], args_ns=a)

        run([PY, script("03_compare_demo_vs_full", "compare_pro.py"),
             "--base", str(base_dir), "--outdir", str(compare_dir), "--dataset_tag", domain], args_ns=a)

        run([PY, script("03_compare_demo_vs_full", "compare_addons.py"),
             "--base", str(base_dir), "--outdir", str(compare_dir)], args_ns=a)

        run([PY, script("03_compare_demo_vs_full", "compare_world_heatmap.py"),
             "--input", str(compare_dir / "metrics_aggregated_deltas_demo_vs_full.csv"),
             "--outdir", str(compare_dir), "--dataset_tag", domain], args_ns=a)

        demo_csv = a.out_root / f"{domain}_demo" / demo_name
        full_csv = a.out_root / f"{domain}_full" / full_name
        # subsample size defaults to the Demo row count inside the script
        trust_cmd = [PY, script("03_compare_demo_vs_full", "compare_trustworthiness.py"),
                     "--full_path", str(full_csv), "--demo_path", str(demo_csv),
                     "--target_col", target, "--n_jobs", str(a.trust_n_jobs),
                     "--outdir", str(compare_dir / "trustworthiness")]
        if a.svc_max_train_n:
            trust_cmd += ["--svc_max_train_n", str(a.svc_max_train_n)]
        run(trust_cmd, allow_missing_input=full_csv, args_ns=a)

        # ICU-patient null pool: every official Demo patient has an ICU stay, so the
        # null is repeated with draws restricted to Full patients with >= 1 ICU stay.
        icustays = a.mimic_root / "MIMIC-IV-Full-2.2" / "icu" / "icustays.csv.gz"
        icu_list = a.out_root / "checks" / "full_icu_subject_ids.csv"
        if not icu_list.exists():
            run([PY, script("01_prepare_data", "make_icu_subject_list.py"),
                 "--icustays", str(icustays), "--out", str(icu_list)],
                allow_missing_input=icustays, args_ns=a)
        trust_icu_cmd = [PY, script("03_compare_demo_vs_full", "compare_trustworthiness.py"),
                         "--full_path", str(full_csv), "--demo_path", str(demo_csv),
                         "--target_col", target, "--n_jobs", str(a.trust_n_jobs),
                         "--null_pool", "icu", "--icu_subjects_path", str(icu_list),
                         "--outdir", str(compare_dir / "trustworthiness_icu")]
        if a.svc_max_train_n:
            trust_icu_cmd += ["--svc_max_train_n", str(a.svc_max_train_n)]
        run(trust_icu_cmd, allow_missing_input=icu_list, args_ns=a)

        run([PY, script("03_compare_demo_vs_full", "compare_rsce_vs_degradation.py"),
             "--rsce_scores", str(full_rsce / "rsce_scores.csv"),
             "--metrics_aggregated", str(full_rsce / "metrics_aggregated.csv"),
             "--outdir", str(a.out_root / f"{domain}_full" / "degradation_analysis"),
             "--dataset_tag", f"{domain}_full"], args_ns=a)

        run([PY, script("03_compare_demo_vs_full", "compare_rsce_vs_degradation.py"),
             "--rsce_scores", str(demo_rsce / "rsce_scores.csv"),
             "--metrics_aggregated", str(demo_rsce / "metrics_aggregated.csv"),
             "--outdir", str(a.out_root / f"{domain}_demo" / "degradation_analysis"),
             "--dataset_tag", f"{domain}_demo"], args_ns=a)

        demo_ppv = a.out_root / f"{domain}_demo" / "ppv" / "ppv_std_per_fold.csv"
        full_ppv = a.out_root / f"{domain}_full" / "ppv" / "ppv_std_per_fold.csv"
        if demo_ppv.exists() and full_ppv.exists():
            run([PY, script("04_ppv", "compare_ppv.py"),
                 "--full_per_fold", str(full_ppv), "--demo_per_fold", str(demo_ppv),
                 "--outdir", str(compare_dir / "ppv")], args_ns=a)
        else:
            log(f"[SKIP] {domain}: missing PPV outputs for demo and/or full; skipping compare_ppv.py.")


def stage_extras(a) -> None:
    banner("STAGE: extras (Table 1, cross-domain synthesis, derived analyses, figures, environment snapshot)")

    ds_args = []
    hosp_demo_csv = a.out_root / "hosp_demo" / "demo_analytic_dataset_mortality_all_admissions.csv"
    hosp_full_csv = a.out_root / "hosp_full" / "full_analytic_dataset_mortality_all_admissions.csv"
    ed_demo_csv = a.out_root / "ed_demo" / "demo_ed_analytic_dataset_admission.csv"
    ed_full_csv = a.out_root / "ed_full" / "full_ed_analytic_dataset_admission.csv"
    if hosp_demo_csv.exists():
        ds_args += ["--dataset", f"hosp_demo={hosp_demo_csv}:label_mortality"]
    if hosp_full_csv.exists():
        ds_args += ["--dataset", f"hosp_full={hosp_full_csv}:label_mortality"]
    if ed_demo_csv.exists():
        ds_args += ["--dataset", f"ed_demo={ed_demo_csv}:label_ed_admit"]
    if ed_full_csv.exists():
        ds_args += ["--dataset", f"ed_full={ed_full_csv}:label_ed_admit"]

    if ds_args:
        run([PY, script("01_prepare_data", "make_table1.py")] + ds_args +
            ["--outdir", str(a.out_root / "table1")], args_ns=a)

    hosp_compare = a.out_root / "compare" / "hosp"
    ed_compare = a.out_root / "compare" / "ed"
    if (hosp_compare / "rsce_comparison_demo_vs_full.csv").exists() and \
       (ed_compare / "rsce_comparison_demo_vs_full.csv").exists():
        run([PY, script("03_compare_demo_vs_full", "synthesize_cross_domain.py"),
             "--hosp_compare_dir", str(hosp_compare), "--ed_compare_dir", str(ed_compare),
             "--outdir", str(a.out_root / "compare" / "cross_domain")], args_ns=a)
    else:
        log("[SKIP] cross-domain synthesis needs compare_pro.py output for BOTH hosp and ed "
            "(run with --run_full first); skipping.")

    # Derived analyses (read finished result files only; no model fitting).
    if all((a.out_root / "compare" / d / pool / "full_reference_metrics.csv").exists()
           for d in ("hosp", "ed") for pool in ("trustworthiness", "trustworthiness_icu")):
        run([PY, script("03_compare_demo_vs_full", "trustworthiness_selection_regret.py"),
             "--results", str(a.out_root)], args_ns=a)
    else:
        log("[SKIP] selection regret needs both trustworthiness pools (all, icu) of both domains; skipping.")
    if all((a.out_root / f"{d}_{s}" / "rsce" / "rsce_scores.csv").exists()
           for d in ("hosp", "ed") for s in ("demo", "full")):
        run([PY, script("03_compare_demo_vs_full", "rsce_rsc_rank_agreement.py"),
             "--results", str(a.out_root)], args_ns=a)
    else:
        log("[SKIP] RSCE_RSC rank agreement needs Demo and Full RSCE outputs of both domains; skipping.")

    # Figures 1 to 5 (need the Full-scale comparison outputs).
    if (a.out_root / "compare" / "cross_domain" / "cross_domain_summary.csv").exists() and \
       all((a.out_root / "compare" / d / "ppv" / "per_model_full_vs_demo_unpaired.csv").exists() and
           (a.out_root / "compare" / d / "_base" / "full_rsce_per_fold.csv").exists()
           for d in ("hosp", "ed")):
        run([PY, script("05_figures", "make_figures.py"),
             "--root", str(a.out_root.parent), "--outdir", str(a.out_root.parent / "figures")], args_ns=a)
    else:
        log("[SKIP] figures need the Full-scale comparison outputs, incl. Results/compare/<track>/_base "
            "(run with --run_full first, or make_compare_base.py); skipping.")

    note = "final Full run (all modules)" if a.run_full and not a.demo_only else "Demo-scale run"
    run([PY, script("capture_environment.py"),
         "--outdir", str(a.out_root), "--note", note], args_ns=a)


STAGES = {
    "prepare": stage_prepare,
    "checks": stage_checks,
    "estimate": stage_estimate,
    "rsce": stage_rsce,
    "ppv": stage_ppv,
    "compare": stage_compare,
    "extras": stage_extras,
}
STAGE_ORDER = ["prepare", "checks", "estimate", "rsce", "ppv", "compare", "extras"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the entire MIMIC-IV Demo-vs-Full audit pipeline end-to-end.")
    p.add_argument("--mimic_root", type=str, default=str(PIPELINE_ROOT.parent / "MIMIC_Dataset"),
                    help="Folder containing MIMIC-IV-Demo-2.2 / MIMIC-IV-Full-2.2 / MIMIC-IV-ED-Demo-2.2 / "
                         "MIMIC-IV-ED-Full-2.2 (default: ../MIMIC_Dataset relative to this script).")
    p.add_argument("--out_root", type=str, default=str(PIPELINE_ROOT.parent / "Results"),
                    help="Where all outputs go (default: ../Results relative to this script).")
    p.add_argument("--demo_only", action="store_true",
                    help="Only run Demo-scale steps (data prep, RSCE, PPV) -- fast, good for a full "
                         "dry-run of the pipeline's plumbing before touching Full-scale data.")
    p.add_argument("--run_full", action="store_true",
                    help="Actually execute the expensive Full-scale RSCE/PPV/compare stages. Without "
                         "this flag, Full-scale data is still PREPARED and feasibility-ESTIMATED, but "
                         "the actual long-running benchmark is not started -- review the estimate first.")
    p.add_argument("--stage", type=str, choices=["all"] + STAGE_ORDER, default="all",
                    help="Run only one stage (for resuming after an interruption). Default: all, in order.")
    p.add_argument("--list", action="store_true", help="Print the stage plan and exit without running anything.")
    p.add_argument("--continue_on_error", action="store_true",
                    help="Keep going after a failed step instead of stopping the whole run.")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chunksize", type=int, default=2_000_000,
                    help="Row-chunk size for streaming Full-scale labevents.csv.gz / vitalsign.csv.gz.")
    p.add_argument("--landmark_hours", type=float, default=1.0,
                    help="prepare_ed.py's --landmark_hours (ED prediction time = ED arrival + this; Demo and Full).")
    p.add_argument("--hosp_landmark_hours", type=float, default=24.0,
                    help="prepare_hosp.py's --landmark_hours (labs up to admittime + this; admissions ended by then "
                         "excluded). 0 = whole-admission labs (leaks for mortality).")
    p.add_argument("--shap_samples", type=int, default=50,
                    help="Passed to every run_rsce.py call (Demo and Full use the same value).")
    p.add_argument("--trust_n_jobs", type=int, default=4,
                    help="compare_trustworthiness.py --n_jobs (worker processes for the null draws).")
    p.add_argument("--exclude_models", type=str, nargs="*", default=[],
                    help="Passed through to run_rsce.py/run_ppv.py's --exclude_models on every Full-scale call.")
    p.add_argument("--svc_max_train_n", type=int, default=20000,
                    help="Passed to run_rsce.py / run_ppv.py / compare_trustworthiness.py on every call "
                         "(Demo and Full; no effect on Demo-sized data). 0 disables the cap.")
    p.add_argument("--no_shap", action="store_true",
                    help="Passes --no-compute_shap to every run_rsce.py call (Demo and Full).")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=None,
                    help="Passed to run_ppv.py's --fast (smaller tree counts -> different models). Default (unset): "
                         "on only for --demo_only smoke tests; off whenever Demo is compared against Full. "
                         "Pass --fast/--no-fast explicitly to force one behavior for both.")

    args = p.parse_args()
    args.mimic_root = Path(args.mimic_root).expanduser().resolve()
    args.out_root = Path(args.out_root).expanduser().resolve()
    return args


def main() -> None:
    a = parse_args()

    order = STAGE_ORDER if a.stage == "all" else [a.stage]

    if a.list:
        print("Planned stages (in order):")
        for s in order:
            print(f"  - {s}")
        print(f"\nmimic_root = {a.mimic_root}")
        print(f"out_root   = {a.out_root}")
        print(f"demo_only  = {a.demo_only}")
        print(f"run_full   = {a.run_full}")
        return

    banner("MIMIC-IV Demo-vs-Full audit pipeline -- full run")
    log(f"mimic_root = {a.mimic_root}")
    log(f"out_root   = {a.out_root}")
    log(f"demo_only  = {a.demo_only}    run_full = {a.run_full}    stage(s) = {order}")
    if not a.mimic_root.exists():
        log(f"[FATAL] --mimic_root does not exist: {a.mimic_root}")
        flush_log(a.out_root)
        sys.exit(2)

    t_start = time.perf_counter()
    try:
        for stage_name in order:
            STAGES[stage_name](a)
    except StepFailed as e:
        banner("PIPELINE STOPPED (a step failed)")
        log(str(e))
        log("Fix the issue above, then re-run with --stage <name> to resume from that stage "
            "(earlier stages' outputs are left in place, not recomputed).")
        flush_log(a.out_root)
        sys.exit(1)
    except KeyboardInterrupt:
        banner("PIPELINE INTERRUPTED (Ctrl+C)")
        flush_log(a.out_root)
        sys.exit(130)

    dt = time.perf_counter() - t_start
    banner(f"PIPELINE FINISHED in {dt/3600.0:.2f} hours ({dt:,.0f}s)")
    log(f"Outputs under: {a.out_root}")
    log(f"Full log appended to: {a.out_root / 'run_full_pipeline.log'}")
    flush_log(a.out_root)


if __name__ == "__main__":
    main()
