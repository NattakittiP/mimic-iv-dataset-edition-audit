#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
Builds Figures 1 to 5 of the MIMIC-IV Demo-vs-Full audit directly from the
finished result files. No number in
any figure is typed by hand: every plotted value, label value and annotation is
read from a file under Results/ (paths in INPUTS below), and the script checks
the files against each other before drawing (see check_inputs()).

Figure 1 study design and cohort; Figure 2 RSCE Full vs Demo and its exact
decomposition; Figure 3 rank reproducibility; Figure 4 prevalence-standardized
PPV; Figure 5 trustworthiness null (recovery of the Full-best model).

Usage (from anywhere):
  python make_figures.py --root "<repository root>" --outdir "<repository root>/figures"

  <repository root> is the folder holding Results/ (and Pipeline/).
  Results/compare/<dom>/_base/ must exist (make_compare_base.py rebuilds it
  from Results/<dom>_{demo,full}/rsce/).

Outputs in --outdir:
  Fig1_study_design_and_cohort.{pdf,png,tiff}
  Fig2_rsce_full_vs_demo_and_decomposition.{pdf,png,tiff}
  Fig3_rank_reproducibility.{pdf,png,tiff}
  Fig4_ppv_std_full_vs_demo.{pdf,png,tiff}
  Fig5_trustworthiness_null_full_best_recovery.{pdf,png,tiff}
  source_data/Fig*_source_data.csv   every plotted number, with its source file
  figure_manifest.json               SHA-256 of every input and output, versions
PDF: vector, TrueType fonts embedded (pdf.fonttype 42). PNG/TIFF: 600 dpi, RGB.
Size: 7.16 in wide (two-column print width), text >= 7 pt at print size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

# --------------------------------------------------------------------------- inputs
INPUTS = {
    "table1": "Results/table1/table1_combined.csv",
    "flow_hosp_full": "Results/hosp_full/full_analytic_dataset_mortality_all_admissions.cohort_flow.json",
    "flow_hosp_demo": "Results/hosp_demo/demo_analytic_dataset_mortality_all_admissions.cohort_flow.json",
    "flow_ed_full": "Results/ed_full/full_ed_analytic_dataset_admission.cohort_flow.json",
    "flow_ed_demo": "Results/ed_demo/demo_ed_analytic_dataset_admission.cohort_flow.json",
    "schema_hosp_full": "Results/hosp_full/rsce/schema.json",
    "schema_hosp_demo": "Results/hosp_demo/rsce/schema.json",
    "schema_ed_full": "Results/ed_full/rsce/schema.json",
    "schema_ed_demo": "Results/ed_demo/rsce/schema.json",
    "cmp_hosp": "Results/compare/hosp/rsce_comparison_demo_vs_full.csv",
    "cmp_ed": "Results/compare/ed/rsce_comparison_demo_vs_full.csv",
    "dec_hosp": "Results/compare/hosp/delta_rsce_decomposition.csv",
    "dec_ed": "Results/compare/ed/delta_rsce_decomposition.csv",
    "rank_hosp": "Results/compare/hosp/rank_agreement.csv",
    "rank_ed": "Results/compare/ed/rank_agreement.csv",
    "sign_hosp": "Results/compare/hosp/sign_test_summary.csv",
    "sign_ed": "Results/compare/ed/sign_test_summary.csv",
    "cross_domain": "Results/compare/cross_domain/cross_domain_summary.csv",
    "fold_hosp_full": "Results/compare/hosp/_base/full_rsce_per_fold.csv",
    "fold_hosp_demo": "Results/compare/hosp/_base/demo_rsce_per_fold.csv",
    "fold_ed_full": "Results/compare/ed/_base/full_rsce_per_fold.csv",
    "fold_ed_demo": "Results/compare/ed/_base/demo_rsce_per_fold.csv",
    "ppv_agg_hosp_full": "Results/hosp_full/ppv/ppv_std_aggregated.csv",
    "ppv_agg_hosp_demo": "Results/hosp_demo/ppv/ppv_std_aggregated.csv",
    "ppv_agg_ed_full": "Results/ed_full/ppv/ppv_std_aggregated.csv",
    "ppv_agg_ed_demo": "Results/ed_demo/ppv/ppv_std_aggregated.csv",
    "ppv_cmp_hosp": "Results/compare/hosp/ppv/per_model_full_vs_demo_unpaired.csv",
    "ppv_cmp_ed": "Results/compare/ed/ppv/per_model_full_vs_demo_unpaired.csv",
}
# Figure 5: trustworthiness nulls, all-patient pool (trustworthiness/) and ICU-patient pool (trustworthiness_icu/)
TW_DIR = {"all": "trustworthiness", "icu": "trustworthiness_icu"}
TW_SFX = {"random": "", "matched": "_prevmatched"}
for _d in ("hosp", "ed"):
    for _pool, _dir in TW_DIR.items():
        INPUTS[f"tw_{_pool}_{_d}_runinfo"] = f"Results/compare/{_d}/{_dir}/run_info.json"
        INPUTS[f"tw_{_pool}_{_d}_fullref"] = f"Results/compare/{_d}/{_dir}/full_reference_metrics.csv"
        for _n, _sfx in TW_SFX.items():
            INPUTS[f"tw_{_pool}_{_d}_{_n}_exp3"] = f"Results/compare/{_d}/{_dir}/exp3_decision_stability{_sfx}.csv"
            INPUTS[f"tw_{_pool}_{_d}_{_n}_long"] = f"Results/compare/{_d}/{_dir}/subsample_metrics_long{_sfx}.csv"

DOMAINS = ("hosp", "ed")
DOMAIN_TITLE = {"hosp": "Hospital mortality (hosp)", "ed": "ED disposition (ED)"}
MODEL_LABEL = {
    "Logistic_L2": "Logistic (L2)", "RandomForest": "Random forest", "ExtraTrees": "Extra trees",
    "GradientBoosting": "Gradient boosting", "SVC_RBF": "SVC (RBF)", "MLP": "MLP", "GaussianNB": "Gaussian NB",
}
# model order in which the run_rsce.py model zoo is defined (used in Fig. 1 text)
ZOO_ORDER = ["Logistic_L2", "RandomForest", "ExtraTrees", "GradientBoosting", "SVC_RBF", "MLP", "GaussianNB"]
WORLD_LABEL = {
    "clean": "clean", "noise_outliers": "noise + outliers", "missingness": "missingness", "shift": "mean shift",
    "surrogate": "surrogate corruption", "nonlinear": "nonlinear distortion", "subgroup_shift": "subgroup shift",
    "prevalence_shift": "prevalence shift", "concept_drift": "concept drift", "label_noise": "label noise",
}

# --------------------------------------------------------------------------- style
# Palette checked with the dataviz validate_palette.js (light surface #ffffff):
#   Full/Demo pair: PASS on every check.  R/S/C/E: PASS on CVD and normal-vision
#   separation, contrast WARN -> every segment set carries a legend and a source table.
C_FULL, C_DEMO = "#2a78d6", "#eb6834"
C_COMP = {"R": "#4a3aa7", "S": "#1baf7a", "C": "#eda100", "E": "#e87ba4"}
INK, INK2, MUTED, GRID, LINE = "#1f1f1f", "#4a4a4a", "#7a7a7a", "#e3e3e3", "#9a9a9a"
TINT_FULL, TINT_DEMO = "#eaf2fc", "#fdeee7"
BOX_FILL, BOX_EDGE = "#f6f6f4", "#bdbdb8"
FS, FS_SMALL, FS_PANEL = 8.0, 7.0, 10.0
W2 = 7.16  # two-column print width, inches


def setup_style() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((n for n in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if n in names), "sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"], "font.size": FS,
        "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS,
        "legend.fontsize": FS_SMALL, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.0, "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True, "savefig.facecolor": "white", "figure.facecolor": "white",
        "mathtext.default": "regular",
    })
    return family


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.0, ha: str = "right", va: str = "bottom") -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va=va, ha=ha)


# --------------------------------------------------------------------------- loading
class Data:
    def __init__(self, root: Path):
        self.root = root
        self.paths = {k: root / v for k, v in INPUTS.items()}
        missing = [str(p) for p in self.paths.values() if not p.exists()]
        if missing:
            raise SystemExit("STOP: missing input files:\n  " + "\n  ".join(missing))
        rd = lambda k: pd.read_csv(self.paths[k], float_precision="round_trip")  # noqa: E731
        rj = lambda k: json.loads(self.paths[k].read_text(encoding="utf-8"))  # noqa: E731
        self.t1 = rd("table1").set_index("tag")
        self.flow = {f"{d}_{s}": rj(f"flow_{d}_{s}") for d in DOMAINS for s in ("full", "demo")}
        self.schema = {f"{d}_{s}": rj(f"schema_{d}_{s}") for d in DOMAINS for s in ("full", "demo")}
        self.cmp = {d: rd(f"cmp_{d}").set_index("model") for d in DOMAINS}
        self.dec = {d: rd(f"dec_{d}").set_index("model") for d in DOMAINS}
        self.rank = {d: rd(f"rank_{d}").iloc[0] for d in DOMAINS}
        self.sign = {d: rd(f"sign_{d}").iloc[0] for d in DOMAINS}
        self.xdom = rd("cross_domain").set_index("domain")
        self.fold = {(d, s): rd(f"fold_{d}_{s}") for d in DOMAINS for s in ("full", "demo")}
        self.ppv_agg = {(d, s): rd(f"ppv_agg_{d}_{s}").set_index("model") for d in DOMAINS for s in ("full", "demo")}
        self.ppv_cmp = {d: rd(f"ppv_cmp_{d}").set_index("model") for d in DOMAINS}
        self.tw_exp3 = {(pl, d, n): rd(f"tw_{pl}_{d}_{n}_exp3").iloc[0]
                        for pl in TW_DIR for d in DOMAINS for n in TW_SFX}
        self.tw_long = {(pl, d, n): rd(f"tw_{pl}_{d}_{n}_long") for pl in TW_DIR for d in DOMAINS for n in TW_SFX}
        self.tw_fullref = {(pl, d): rd(f"tw_{pl}_{d}_fullref").set_index("model") for pl in TW_DIR for d in DOMAINS}
        self.tw_runinfo = {(pl, d): rj(f"tw_{pl}_{d}_runinfo") for pl in TW_DIR for d in DOMAINS}


def check_inputs(D: Data) -> List[str]:
    """Cross-file consistency checks; any failure stops the script."""
    ok: List[str] = []
    tol = 1e-9

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise SystemExit(f"STOP (input check failed): {msg}")
        ok.append(msg)

    for d in DOMAINS:
        models = set(D.cmp[d].index)
        need(len(models) == 7, f"{d}: 7 models in rsce_comparison")
        for s in ("full", "demo"):
            f = D.fold[(d, s)]
            need(set(f["model"]) == models and f.groupby("model").size().eq(15).all() and f["RSCE_fold"].notna().all(),
                 f"{d}/{s}: per-fold file has 15 non-missing folds for each of the 7 models")
            m = f.groupby("model")["RSCE_fold"].mean()
            need(float((m - D.cmp[d][f"RSCE_full_{s}"]).abs().max()) < 1e-12,
                 f"{d}/{s}: mean of per-fold RSCE equals RSCE_full_{s} in rsce_comparison")
            sch, t1 = D.schema[f"{d}_{s}"], D.t1.loc[f"{d}_{s}"]
            need(int(sch["n_samples"]) == int(t1["n_rows"]) == int(D.flow[f"{d}_{s}"]["n_rows_final"]),
                 f"{d}/{s}: n rows agree in schema.json, table1 and cohort_flow")
            need(int(t1["n_label_positive"]) + int(t1["n_label_negative"]) == int(t1["n_rows"]),
                 f"{d}/{s}: positives + negatives = rows")
            need(sch["folds"] == 5 and sch["repeats"] == 3 and sch["cv_mode"] == "group" and sch["group_col"] == "subject_id",
                 f"{d}/{s}: 5x3 patient-grouped CV in schema.json")
            a = D.ppv_agg[(d, s)]
            need(set(a.index) == models, f"{d}/{s}: PPV file has the same 7 models")
            col = "ppv_std_pooled_full" if s == "full" else "ppv_std_pooled_demo"
            need(float((a["ppv_std_pooled_mean"] - D.ppv_cmp[d][col].reindex(a.index)).abs().max()) < 1e-12,
                 f"{d}/{s}: PPV_std pooled mean equals the value in the Full-vs-Demo PPV comparison")
            need(float((a["pi_ref"] - D.t1.loc[f"{d}_full", "label_prevalence"]).abs().max()) < 1e-12,
                 f"{d}/{s}: PPV reference prevalence equals the Full prevalence in table1")
        dec = D.dec[d]
        tot = dec[["contrib_R", "contrib_S", "contrib_C", "contrib_E"]].sum(axis=1)
        need(float((tot - D.cmp[d]["RSCE_full_delta"].reindex(tot.index)).abs().max()) < 1e-12,
             f"{d}: R+S+C+E contributions sum to RSCE_full_delta for every model")
        need(float(abs(dec["contrib_R"].mean() - D.xdom.loc[d, "mean_contrib_R"])) < 1e-12,
             f"{d}: mean R contribution equals cross_domain_summary")
        need(float(abs(D.cmp[d]["RSCE_full_delta"].mean() - D.xdom.loc[d, "mean_delta_rsce_full_minus_demo"])) < 1e-12,
             f"{d}: mean delta equals cross_domain_summary")
        rf = D.cmp[d]["RSCE_full_full"].rank(ascending=False)
        rdm = D.cmp[d]["RSCE_full_demo"].rank(ascending=False)
        need(bool((rf == D.cmp[d]["rank_full"]).all() and (rdm == D.cmp[d]["rank_demo"]).all()),
             f"{d}: rank columns equal the ranks of the RSCE values (no ties)")
        rho = float(np.corrcoef(rf, rdm)[0, 1])
        need(abs(rho - float(D.rank[d]["spearman_rank"])) < 1e-9, f"{d}: Spearman of the ranks equals rank_agreement.csv")
        need(int(D.sign[d]["positive"]) == int((D.cmp[d]["RSCE_full_delta"] > 0).sum()),
             f"{d}: sign-test positive count equals the number of positive deltas")
        need(D.cmp[d]["RSCE_full_full"].idxmax() == D.xdom.loc[d, "best_model_full"]
             and D.cmp[d]["RSCE_full_demo"].idxmax() == D.xdom.loc[d, "best_model_demo"],
             f"{d}: best models equal cross_domain_summary")
    for k in ("hosp_demo", "ed_demo"):
        need(abs(D.t1.loc[k, "label_prevalence"] - D.t1.loc[k, "n_label_positive"] / D.t1.loc[k, "n_rows"]) < tol,
             f"{k}: prevalence = positives / rows")
    for d in DOMAINS:
        need(D.schema[f"{d}_full"]["worlds"] == D.schema[f"{d}_demo"]["worlds"]
             and D.schema[f"{d}_full"]["scoring"] == D.schema[f"{d}_demo"]["scoring"]
             and D.schema[f"{d}_full"]["model_params"] == D.schema[f"{d}_demo"]["model_params"],
             f"{d}: Demo and Full runs used the same worlds, scoring and model parameters")
    need(D.schema["hosp_full"]["worlds"] == D.schema["ed_full"]["worlds"], "hosp and ED used the same 10 worlds")
    # ---- Figure 5 inputs
    mode_name = {"random": "random", "matched": "prevalence_matched"}
    for pl in TW_DIR:
        for d in DOMAINS:
            ri = D.tw_runinfo[(pl, d)]
            need((ri.get("null_pool", "all") == pl), f"{pl}/{d}: run_info null_pool is '{pl}'")
            fbest = D.tw_fullref[(pl, d)]["AUROC"].sort_values(ascending=False).index[0]
            for n in TW_SFX:
                e, L = D.tw_exp3[(pl, d, n)], D.tw_long[(pl, d, n)]
                need(int(e["n_null_valid"]) == 1000 and len(L) == 7000 and L["run_id"].nunique() == 1000
                     and bool((L["null_mode"] == mode_name[n]).all()) and int(L["AUROC"].isna().sum()) == 0,
                     f"{pl}/{d}/{n}: 1,000 valid draws x 7 models, no NaN AUROC")
                need(e["full_best_model"] == fbest, f"{pl}/{d}/{n}: exp3 Full-best equals argmax of full_reference AUROC")
                # same rule as compare_trustworthiness.topk_set: sort the draw's 7 rows by AUROC (descending,
                # pandas default sort) and take the first k models; ties are resolved exactly as there
                k1 = k3 = 0
                for _, g in L.sort_values(["run_id", "model"]).groupby("run_id"):
                    order_ = g.dropna(subset=["AUROC"]).sort_values("AUROC", ascending=False)["model"].tolist()
                    k1 += int(order_[0] == fbest)
                    k3 += int(fbest in order_[:3])
                for kk, col in ((k1, "subsample_P(best_matches_full)"), (k3, "subsample_P(full_best_in_top3)")):
                    need(abs(kk / 1000 - float(e[col])) < 1e-12, f"{pl}/{d}/{n}: {col} recomputed from the null draws")
                for kk, pre in ((k1, "subsample_P"), (k3, "subsample_P(full_best_in_top3)")):
                    lo, hi = wilson(kk, 1000)
                    need(abs(lo - float(e[f"{pre}_wilson95_low"])) < 1e-9 and abs(hi - float(e[f"{pre}_wilson95_high"])) < 1e-9,
                         f"{pl}/{d}/{n}: Wilson 95% interval of {pre} recomputed")
    return ok


def wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def save(fig, outdir: Path, stem: str, outputs: List[Path]) -> None:
    for ext in ("pdf", "png", "tiff"):
        p = outdir / f"{stem}.{ext}"
        kw: Dict = {"dpi": 600}
        if ext == "pdf":
            kw["metadata"] = {"CreationDate": None, "Creator": "make_figures.py", "Producer": None}
        elif ext == "png":
            kw["metadata"] = {"Software": None}
        else:
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, format=ext, **kw)
        if ext in ("png", "tiff"):  # flatten to RGB (no alpha channel), keep 600 dpi
            from PIL import Image
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            if ext == "png":
                rgb.save(p, format="PNG", dpi=(600, 600), optimize=True)
            else:
                rgb.save(p, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
        outputs.append(p)
    plt.close(fig)


# =========================================================================== FIGURE 1
def _box(ax, x, y, w, h, fc=BOX_FILL, ec=BOX_EDGE, lw=0.7, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, mutation_aspect=1))


def _arrow(ax, p0, p1, rad=0.0, lw=0.9, color=INK2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=3))


def _lines(ax, x, y, lines, dy, **kw):
    """lines: list of (text, style dict) drawn top-down from y."""
    for i, (t, st) in enumerate(lines):
        ax.text(x, y - i * dy, t, va="top", **{**kw, **st})


def fig1(D: Data, outdir: Path, outputs: List[Path], src_rows: List[Dict]) -> None:
    t1, flow, sch = D.t1, D.flow, D.schema
    s0 = sch["hosp_full"]
    worlds = s0["worlds"]
    w = {k: round(float(v), 3) for k, v in s0["scoring"]["weights_normalized"].items()}
    n_feat = {d: len(sch[f"{d}_full"]["numeric_cols"]) + len(sch[f"{d}_full"]["categorical_cols"]) for d in DOMAINS}
    hosp_labs = sum(c.startswith("lab_") for c in s0["numeric_cols"])
    hosp_cat = len(s0["categorical_cols"])
    edn = sch["ed_full"]["numeric_cols"]
    ed_triage = sum(c.startswith("triage_") for c in edn)
    ed_vital = sum(c.startswith("vital_") for c in edn)
    ed_meds = sum(c in ("n_home_meds", "n_ed_meds") for c in edn)
    if n_feat["hosp"] != hosp_labs + 1 + hosp_cat or "anchor_age" not in s0["numeric_cols"]:
        raise SystemExit("STOP: hosp feature groups do not add up")
    if n_feat["ed"] != ed_triage + ed_vital + ed_meds + 1 + 1 or sch["ed_full"]["categorical_cols"] != ["gender"]:
        raise SystemExit("STOP: ED feature groups do not add up")
    lm = {d: flow[f"{d}_full"]["landmark_hours"] for d in DOMAINS}
    if any(flow[f"{d}_demo"]["landmark_hours"] != lm[d] for d in DOMAINS):
        raise SystemExit("STOP: Demo and Full landmarks differ")
    n_models = len(s0["model_params"])
    n_folds = s0["folds"] * s0["repeats"]

    fig = plt.figure(figsize=(W2, 7.4))
    # panel a spans the full width; panels b and c have their own margins (long y labels in b)
    gsa = fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.995, bottom=0.465)
    gsb = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], left=0.215, right=0.985, top=0.400, bottom=0.063,
                           wspace=0.32)
    ax = fig.add_subplot(gsa[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)
    ax.axis("off")
    ax.text(0.2, 56.9, "a", fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")

    gapx = 2.9
    widths = {"src": 10.6, "coh": 26.6, "bm": 29.6, "an": 22.4}
    cols, x = {}, 1.0
    for k in ("src", "coh", "bm", "an"):
        cols[k] = (x, widths[k])
        x += widths[k] + gapx
    heads = {"src": ("Data releases", "PhysioNet, as released"),
             "coh": ("Cohort construction", "same script for Demo and Full"),
             "bm": ("Identical benchmark", "same settings for Demo and Full"),
             "an": ("Analyses", "Demo compared with Full")}
    for k, (x0, wd) in cols.items():
        ax.text(x0 + wd / 2, 53.9, heads[k][0], ha="center", va="center", fontsize=FS, fontweight="bold")
        ax.text(x0 + wd / 2, 51.4, heads[k][1], ha="center", va="center", fontsize=FS_SMALL, color=MUTED)
    dy = 2.5
    row = {"hosp": (26.5, 23.0), "ed": (0.6, 23.0)}  # y0, height
    top = row["hosp"][0] + row["hosp"][1]

    # --- column 1: releases with Demo / Full chips
    rel_name = {"hosp": "MIMIC-IV\nv2.2 (hosp)", "ed": "MIMIC-IV-ED\nv2.2"}
    for d in DOMAINS:
        y0, h = row[d]
        x0, wd = cols["src"]
        _box(ax, x0, y0, wd, h)
        ax.text(x0 + wd / 2, y0 + h - 1.6, rel_name[d], ha="center", va="top", fontsize=FS, fontweight="bold",
                linespacing=1.15)
        for j, (lab, ec, fc) in enumerate((("Demo", C_DEMO, TINT_DEMO), ("Full", C_FULL, TINT_FULL))):
            cy = y0 + 10.0 - j * 6.4
            ax.add_patch(FancyBboxPatch((x0 + 1.4, cy - 2.3), wd - 2.8, 4.6, boxstyle="round,pad=0,rounding_size=0.9",
                                        fc=fc, ec=ec, lw=1.2, zorder=2))
            ax.text(x0 + wd / 2, cy, lab, ha="center", va="center", fontsize=FS, zorder=3)

    # --- column 2: cohort definitions (numbers read from schema.json / cohort_flow.json)
    coh_txt = {
        "hosp": [
            ("Hospital mortality (hosp)", {"fontweight": "bold"}),
            ("Unit: adult admission", {}),
            ("Outcome: in-hospital death", {}),
            (f"Landmark: admission + {lm['hosp']:g} h", {}),
            (f"{n_feat['hosp']} features: {hosp_labs} lab medians,", {}),
            (f"   age, {hosp_cat} demographic and", {}),
            ("   admission fields", {}),
        ],
        "ed": [
            ("ED disposition (ED)", {"fontweight": "bold"}),
            ("Unit: ED stay", {}),
            ("Outcome: hospital encounter", {}),
            ("   (admitted or observation)", {}),
            (f"Landmark: ED arrival + {lm['ed']:g} h", {}),
            (f"{n_feat['ed']} features: {ed_triage} triage,", {}),
            (f"   {ed_vital} vital-sign, {ed_meds} medication", {}),
            ("   counts, age, sex", {}),
        ],
    }
    for d in DOMAINS:
        y0, h = row[d]
        x0, wd = cols["coh"]
        _box(ax, x0, y0, wd, h)
        _lines(ax, x0 + 1.2, y0 + h - 1.4, coh_txt[d], dy, fontsize=FS)
        _arrow(ax, (cols["src"][0] + cols["src"][1] + 0.3, y0 + h / 2), (x0 - 0.3, y0 + h / 2))
    ax.text(cols["coh"][0] + cols["coh"][1] / 2, (row["ed"][0] + row["ed"][1] + row["hosp"][0]) / 2,
            "units that ended before the landmark are excluded", ha="center", va="center", fontsize=FS_SMALL, color=MUTED,
            style="italic")

    # --- column 3: benchmark (all values from schema.json)
    x0, wd = cols["bm"]
    yb0 = row["ed"][0]
    _box(ax, x0, yb0, wd, top - yb0)
    wl = [WORLD_LABEL[wd_["kind"]] for wd_ in worlds]
    zoo = [MODEL_LABEL[m] for m in ZOO_ORDER if m in s0["model_params"]]
    if len(zoo) != n_models:
        raise SystemExit("STOP: model zoo in schema.json differs from the figure's model list")
    bm_lines = [
        ("Cross-validation", {"fontweight": "bold"}),
        (f"{s0['folds']} folds × {s0['repeats']} repeats = {n_folds} test folds,", {}),
        (f"grouped by patient, seed {s0['seed']}", {}),
        (f"{n_models} fixed models, no tuning", {"fontweight": "bold"}),
        (", ".join(zoo[:2]) + ",", {}),
        (", ".join(zoo[2:4]) + ",", {}),
        (", ".join(zoo[4:]), {}),
        (f"{len(worlds)} worlds, test fold only", {"fontweight": "bold"}),
        (", ".join(wl[:3]) + ",", {}),
        (", ".join(wl[3:5]) + ",", {}),
        (", ".join(wl[5:7]) + ",", {}),
        (", ".join(wl[7:9]) + ",", {}),
        (", ".join(wl[9:]), {}),
        ("Score", {"fontweight": "bold"}),
        (f"RSCE = {w['R']:g} R + {w['S']:g} S + {w['C']:g} C + {w['E']:g} E", {}),
        ("R clean AUROC · S AUROC retention", {"color": INK2}),
        ("C calibration stability · E SHAP stability", {"color": INK2}),
    ]
    yy = top - 1.4
    for i, (t, st) in enumerate(bm_lines):
        if st.get("fontweight") == "bold" and i > 0:
            yy -= 0.75
        ax.text(x0 + 1.2, yy, t, va="top", fontsize=FS, **st)
        yy -= dy
    if yy < yb0:
        raise SystemExit("STOP: benchmark text overflows its box")
    for d in DOMAINS:
        y0, h = row[d]
        _arrow(ax, (cols["coh"][0] + cols["coh"][1] + 0.3, y0 + h / 2), (x0 - 0.3, y0 + h / 2))

    # --- column 4: analyses
    x0, wd = cols["an"]
    an = [
        ("Demo vs Full RSCE", ["per-model difference,", "R/S/C/E decomposition,", "rank agreement"]),
        ("Standardised PPV", ["PPV at sensitivity 0.80,", "Full prevalence as", "reference"]),
        ("Trustworthiness null", ["1,000 Demo-sized Full", "samples: random and", "prevalence-matched"]),
    ]
    # each box is sized to its text (title + 3 lines); boxes are spread evenly over the benchmark box height
    n_lines = 1 + max(len(bd) for _, bd in an)
    hbox = 1.2 + (n_lines - 1) * dy + 1.75 + 1.2
    gap = (top - yb0 - 3 * hbox) / 2
    bm_right = cols["bm"][0] + cols["bm"][1]
    for i, (title, body) in enumerate(an):
        y0 = top - (i + 1) * hbox - i * gap
        _box(ax, x0, y0, wd, hbox)
        ax.text(x0 + 1.1, y0 + hbox - 1.2, title, va="top", fontsize=FS, fontweight="bold")
        _lines(ax, x0 + 1.1, y0 + hbox - 1.2 - dy, [(b, {}) for b in body], dy, fontsize=FS)
        _arrow(ax, (bm_right + 0.3, y0 + hbox / 2), (x0 - 0.3, y0 + hbox / 2))

    # ------------------------------------------------------------------ panel b: counts
    axb = fig.add_subplot(gsb[0, 0])
    cats = {
        "hosp": [("Admissions analysed", "n_rows"), ("Patients", "n_unique_patients"),
                 ("Deaths", "n_label_positive"), ("Survivors", "n_label_negative")],
        "ed": [("ED stays analysed", "n_rows"), ("Patients", "n_unique_patients"),
               ("Admitted or observation", "n_label_positive"), ("Not admitted", "n_label_negative")],
    }
    minority = {"hosp": "n_label_positive", "ed": "n_label_negative"}
    # bar pair per row: Full above, Demo below, with a clear gap so the Demo value label
    # (drawn to the right of the shorter Demo bar, i.e. under the Full bar) does not touch the Full bar
    bh, off, step, lab_dn = 0.28, 0.22, 1.25, 0.07
    y = 0.0
    yt, yl, heads_y = [], [], {}
    for d in DOMAINS:
        heads_y[d] = y
        y -= 0.95
        for lab, col in cats[d]:
            vf, vd = float(t1.loc[f"{d}_full", col]), float(t1.loc[f"{d}_demo", col])
            is_min = col == minority[d]
            if is_min:
                axb.add_patch(Rectangle((1, y - step / 2), 1e8, step, fc="#fff3c4", ec="none", zorder=0))
            axb.barh(y + off, vf, height=bh, left=1, color=C_FULL, zorder=2, lw=0)
            axb.barh(y - off, vd, height=bh, left=1, color=C_DEMO, zorder=2, lw=0)
            axb.text(vf * 1.25, y + off, fmt_int(vf), va="center", fontsize=FS_SMALL, color=INK2)
            axb.text(vd * 1.25, y - off - lab_dn, fmt_int(vd), va="center", fontsize=FS if is_min else FS_SMALL,
                     color=INK, fontweight="bold" if is_min else "normal")
            yt.append(y)
            yl.append(lab)
            src_rows += [{"figure": "Fig1b", "domain": d, "quantity": lab, "release": "Full", "value": vf,
                          "source": INPUTS["table1"] + f" [{d}_full, {col}]"},
                         {"figure": "Fig1b", "domain": d, "quantity": lab, "release": "Demo", "value": vd,
                          "source": INPUTS["table1"] + f" [{d}_demo, {col}]"}]
            y -= step
        y -= 0.3
    axb.set_xscale("log")
    axb.set_xlim(1, 3e6)
    axb.set_ylim(y + step - 0.25 - 0.1, 0.45)
    axb.set_yticks(yt, yl)
    trans = matplotlib.transforms.blended_transform_factory(fig.transFigure, axb.transData)
    for d in DOMAINS:
        axb.text(0.012, heads_y[d] - 0.12, DOMAIN_TITLE[d], transform=trans, ha="left", va="center", fontsize=FS,
                 fontweight="bold")
    axb.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6], ["1", "10", "10²", "10³", "10⁴", "10⁵", "10⁶"])
    axb.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axb.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axb.set_axisbelow(True)
    axb.set_xlabel("Count (log scale)")
    axb.spines["left"].set_visible(False)
    axb.text(0.012, 1.035, "b", transform=matplotlib.transforms.blended_transform_factory(fig.transFigure, axb.transAxes),
             fontsize=FS_PANEL, fontweight="bold", ha="left", va="bottom")
    leg = [Patch(fc=C_FULL, label="Full"), Patch(fc=C_DEMO, label="Demo"),
           Patch(fc="#fff3c4", ec="none", label="minority outcome class")]
    axb.legend(handles=leg, loc="lower center", frameon=False, ncol=3, fontsize=FS_SMALL, handlelength=1.2,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1)

    # ------------------------------------------------------------------ panel c: prevalence
    sub = gsb[0, 1].subgridspec(2, 1, hspace=1.35)
    for i, d in enumerate(DOMAINS):
        axc = fig.add_subplot(sub[i])
        pf = 100 * float(t1.loc[f"{d}_full", "label_prevalence"])
        pd_ = 100 * float(t1.loc[f"{d}_demo", "label_prevalence"])
        axc.plot([pf, pd_], [0, 0], color=LINE, lw=1.6, zorder=1, solid_capstyle="butt")
        axc.scatter([pf], [0], s=42, color=C_FULL, zorder=3, edgecolor="white", lw=0.8)
        axc.scatter([pd_], [0], s=42, color=C_DEMO, zorder=3, edgecolor="white", lw=0.8)
        axc.text(pf, 0.55, f"Full\n{pf:.2f}%", ha="center", va="bottom", fontsize=FS_SMALL, color=INK2, linespacing=1.0)
        axc.text(pd_, 0.55, f"Demo\n{pd_:.2f}%", ha="center", va="bottom", fontsize=FS_SMALL, color=INK2,
                 linespacing=1.0)
        axc.set_xlim(*((0, 8) if d == "hosp" else (0, 100)))
        axc.set_ylim(-0.6, 1.95)
        axc.set_yticks([])
        axc.spines["left"].set_visible(False)
        axc.grid(axis="x", color=GRID, lw=0.5)
        axc.set_axisbelow(True)
        outcome = "in-hospital death" if d == "hosp" else "admitted or observation"
        axc.set_title(f"{'hosp' if d == 'hosp' else 'ED'}: {outcome}", fontsize=FS, loc="left", pad=3)
        axc.set_xlabel("Outcome prevalence (%)", fontsize=FS_SMALL, labelpad=1.5)
        if i == 0:
            axc.text(-0.10, 1.40, "c", transform=axc.transAxes, fontsize=FS_PANEL, fontweight="bold", ha="left",
                     va="bottom")
        src_rows += [{"figure": "Fig1c", "domain": d, "quantity": "prevalence_percent", "release": "Full", "value": pf,
                      "source": INPUTS["table1"] + f" [{d}_full, label_prevalence]"},
                     {"figure": "Fig1c", "domain": d, "quantity": "prevalence_percent", "release": "Demo", "value": pd_,
                      "source": INPUTS["table1"] + f" [{d}_demo, label_prevalence]"}]
    save(fig, outdir, "Fig1_study_design_and_cohort", outputs)


# =========================================================================== FIGURE 2
def fig2(D: Data, outdir: Path, outputs: List[Path], src_rows: List[Dict]) -> None:
    fig = plt.figure(figsize=(W2, 6.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.50, wspace=0.66,
                          left=0.135, right=0.915, top=0.925, bottom=0.115)
    rng = np.random.default_rng(0)  # jitter only (vertical position of fold dots); values are not altered
    for j, d in enumerate(DOMAINS):
        c = D.cmp[d].sort_values("RSCE_full_full", ascending=False)
        order = list(c.index)
        ys = {m: i for i, m in enumerate(order)}
        ax = fig.add_subplot(gs[0, j])
        for s, col, off in (("full", C_FULL, 0.17), ("demo", C_DEMO, -0.17)):
            f = D.fold[(d, s)]
            yy = np.array([ys[m] for m in f["model"]]) - off + rng.uniform(-0.07, 0.07, len(f))
            ax.scatter(f["RSCE_fold"], yy, s=5, color=col, alpha=0.38, lw=0, zorder=2)
            for _, r in f.iterrows():
                src_rows.append({"figure": f"Fig2{'ab'[j]}_folds", "domain": d, "model": r["model"], "release": s,
                                 "fold": int(r["fold"]), "value": float(r["RSCE_fold"]),
                                 "source": INPUTS[f"fold_{d}_{s}"]})
        for m in order:
            vf, vd = c.loc[m, "RSCE_full_full"], c.loc[m, "RSCE_full_demo"]
            ax.plot([vd, vf], [ys[m], ys[m]], color=LINE, lw=1.4, zorder=3, solid_capstyle="butt")
            ax.scatter([vf], [ys[m]], s=34, color=C_FULL, zorder=4, edgecolor="white", lw=0.8)
            ax.scatter([vd], [ys[m]], s=34, color=C_DEMO, zorder=4, edgecolor="white", lw=0.8)
            ax.text(1.003, ys[m], f"{c.loc[m, 'RSCE_full_delta']:.3f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=FS_SMALL, color=INK2)
            src_rows += [{"figure": f"Fig2{'ab'[j]}", "domain": d, "model": m, "release": "Full", "value": float(vf),
                          "source": INPUTS[f"cmp_{d}"] + " [RSCE_full_full]"},
                         {"figure": f"Fig2{'ab'[j]}", "domain": d, "model": m, "release": "Demo", "value": float(vd),
                          "source": INPUTS[f"cmp_{d}"] + " [RSCE_full_demo]"},
                         {"figure": f"Fig2{'ab'[j]}", "domain": d, "model": m, "release": "delta", "value":
                          float(c.loc[m, "RSCE_full_delta"]), "source": INPUTS[f"cmp_{d}"] + " [RSCE_full_delta]"}]
        ax.text(1.003, -0.85, "ΔRSCE", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=FS_SMALL, color=INK2, fontweight="bold")
        ax.set_yticks(range(len(order)), [MODEL_LABEL[m] for m in order])
        ax.set_ylim(len(order) - 0.45, -1.25)
        ax.set_xlim(0.55, 1.0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("RSCE  (small dots: 15 folds; large dots: mean)")
        sg = D.sign[d]
        if int(sg["negative"]) != 0 or int(sg["zero"]) != 0:
            raise SystemExit("STOP: the Fig. 2 title assumes no model is higher on Demo")
        ax.set_title(f"{DOMAIN_TITLE[d]}: RSCE\n{int(sg['positive'])}/{int(sg['effective_n'])} models lower on Demo; "
                     f"sign test p = {float(sg['p_value_two_sided_binomtest']):.4f}", fontsize=FS, loc="left", pad=4)
        panel_letter(ax, "ab"[j], x=-0.36, y=1.02, ha="left")
        # Full/Demo legend centred in the empty band at the top of each of panels a and b
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_FULL, label="Full", ms=5),
                           Line2D([], [], marker="o", ls="", color=C_DEMO, label="Demo", ms=5)],
                  loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=FS_SMALL,
                  handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=0.8)

        # ---- decomposition
        axd = fig.add_subplot(gs[1, j])
        dec = D.dec[d].reindex(order)
        rows = order + ["__mean__"]
        yd = {m: i for i, m in enumerate(order)}
        yd["__mean__"] = len(order) + 0.35
        mean_contrib = {k: float(D.xdom.loc[d, f"mean_contrib_{k}"]) for k in "RSCE"}
        for m in rows:
            vals = mean_contrib if m == "__mean__" else {k: float(dec.loc[m, f"contrib_{k}"]) for k in "RSCE"}
            pos, neg = 0.0, 0.0
            for k in "RSCE":
                v = vals[k]
                if v >= 0:
                    axd.barh(yd[m], v, left=pos, height=0.62, color=C_COMP[k], ec="white", lw=0.5, zorder=2)
                    pos += v
                else:
                    axd.barh(yd[m], v, left=neg, height=0.62, color=C_COMP[k], ec="white", lw=0.5, zorder=2)
                    neg += v
            tot = sum(vals.values())
            axd.plot([tot, tot], [yd[m] - 0.40, yd[m] + 0.40], color=INK, lw=1.0, zorder=4)
            axd.text(1.003, yd[m], f"{tot:.3f}", transform=axd.get_yaxis_transform(), va="center", ha="left",
                     fontsize=FS_SMALL, color=INK2, fontweight="bold" if m == "__mean__" else "normal")
            for k in "RSCE":
                src_rows.append({"figure": f"Fig2{'cd'[j]}", "domain": d, "model": "mean_of_7" if m == "__mean__" else m,
                                 "component": k, "value": vals[k],
                                 "source": (INPUTS["cross_domain"] + f" [mean_contrib_{k}]") if m == "__mean__"
                                 else INPUTS[f"dec_{d}"] + f" [contrib_{k}]"})
        axd.axvline(0, color=INK2, lw=0.6, zorder=3)
        axd.axhline(len(order) - 0.33, color=GRID, lw=0.8)
        axd.set_yticks([yd[m] for m in rows], [MODEL_LABEL[m] for m in order] + ["Mean of 7"])
        lab = axd.get_yticklabels()[-1]
        lab.set_fontweight("bold")
        axd.set_ylim(yd["__mean__"] + 0.55, -1.4)
        axd.set_xlim(-0.022, 0.19)
        axd.grid(axis="x", color=GRID, lw=0.5)
        axd.set_axisbelow(True)
        axd.spines["left"].set_visible(False)
        axd.set_xlabel("Weighted contribution to ΔRSCE (Full − Demo)")
        share_r = mean_contrib["R"] / float(D.xdom.loc[d, "mean_delta_rsce_full_minus_demo"])
        axd.set_title(f"{DOMAIN_TITLE[d]}: ΔRSCE by component\nR carries {100 * share_r:.0f}% of the mean "
                      f"difference", fontsize=FS, loc="left", pad=4)
        axd.text(1.003, -1.0, "Σ", transform=axd.get_yaxis_transform(), ha="left", va="center", fontsize=FS_SMALL,
                 color=INK2, fontweight="bold")
        panel_letter(axd, "cd"[j], x=-0.36, y=1.02, ha="left")
    w = D.schema["hosp_full"]["scoring"]["weights_normalized"]
    handles = [Patch(fc=C_COMP["R"], label=f"R  clean AUROC (×{float(w['R']):.1f})"),
               Patch(fc=C_COMP["S"], label=f"S  AUROC retention (×{float(w['S']):.1f})"),
               Patch(fc=C_COMP["C"], label=f"C  calibration stability (×{float(w['C']):.1f})"),
               Patch(fc=C_COMP["E"], label=f"E  SHAP stability (×{float(w['E']):.1f})"),
               Line2D([], [], color=INK, lw=1.0, label="Σ = ΔRSCE")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FS_SMALL,
               bbox_to_anchor=(0.5, -0.004), handlelength=1.1, columnspacing=1.6, handletextpad=0.4)
    save(fig, outdir, "Fig2_rsce_full_vs_demo_and_decomposition", outputs)


# =========================================================================== FIGURE 3
def _num(x: float, nd: int) -> str:
    """Fixed decimals with a typographic minus sign."""
    return f"{x:.{nd}f}".replace("-", "−")


def fig3(D: Data, outdir: Path, outputs: List[Path], src_rows: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.78, bottom=0.10, wspace=0.02)
    # slope between x = 0 (Full) and x = 1 (Demo); labels need ~1.3 in on each side of a 3.58 in panel
    frac0, frac1 = 0.405, 0.595
    span = 1.0 / (frac1 - frac0)
    xl = (-frac0 * span, (1 - frac0) * span)
    headers = []  # (axis, header texts, row-1 label texts): headers are centred after layout, below
    for j, (ax, d) in enumerate(zip(axes, DOMAINS)):
        c = D.cmp[d]
        row1 = []
        best_f, best_d = c["RSCE_full_full"].idxmax(), c["RSCE_full_demo"].idxmax()
        n = len(c)
        for x in (0, 1):
            ax.plot([x, x], [1, n], color=GRID, lw=0.8, zorder=0)
        for m in c.index:
            rf, rd_ = float(c.loc[m, "rank_full"]), float(c.loc[m, "rank_demo"])
            if m == best_f:
                col, lw, z = C_FULL, 2.0, 4
            elif m == best_d:
                col, lw, z = C_DEMO, 2.0, 4
            else:
                col, lw, z = LINE, 1.1, 2
            ax.plot([0, 1], [rf, rd_], color=col, lw=lw, zorder=z, solid_capstyle="round")
            ax.scatter([0, 1], [rf, rd_], s=22, color=col, zorder=z + 1, edgecolor="white", lw=0.7)
            bold = "bold" if m in (best_f, best_d) else "normal"
            tl = ax.text(-0.06, rf, f"{MODEL_LABEL[m]}  {_num(c.loc[m, 'RSCE_full_full'], 4)}", ha="right",
                         va="center", fontsize=FS, fontweight=bold)
            tr = ax.text(1.06, rd_, f"{_num(c.loc[m, 'RSCE_full_demo'], 4)}  {MODEL_LABEL[m]}", ha="left",
                         va="center", fontsize=FS, fontweight=bold)
            row1 += [t for t, r in ((tl, rf), (tr, rd_)) if r == 1.0]
            src_rows.append({"figure": f"Fig3{'ab'[j]}", "domain": d, "model": m, "rank_full": rf, "rank_demo": rd_,
                             "RSCE_full": float(c.loc[m, "RSCE_full_full"]),
                             "RSCE_demo": float(c.loc[m, "RSCE_full_demo"]), "source": INPUTS[f"cmp_{d}"]})
        hdr = [ax.text(-0.06, 0.0, "Full: RSCE, rank 1 at top", ha="right", va="center", fontsize=FS_SMALL,
                       color=MUTED),
               ax.text(1.06, 0.0, "Demo: RSCE, rank 1 at top", ha="left", va="center", fontsize=FS_SMALL,
                       color=MUTED)]
        headers.append((ax, hdr, row1))
        ax.set_xlim(*xl)
        ax.set_ylim(n + 0.45, -0.25)
        ax.axis("off")
        rk = D.rank[d]
        ax.set_title(f"{DOMAIN_TITLE[d]}\nSpearman ρ = {_num(float(rk['spearman_rank']), 3)}, "
                     f"Kendall τ = {_num(float(rk['kendall_rank']), 3)} ({int(rk['n_models'])} models)",
                     fontsize=FS, pad=6)
        ax.text(0.0, 1.20, "ab"[j], transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", ha="left",
                va="bottom")
        src_rows.append({"figure": f"Fig3{'ab'[j]}", "domain": d, "model": "_agreement_",
                         "spearman": float(rk["spearman_rank"]), "kendall": float(rk["kendall_rank"]),
                         "source": INPUTS[f"rank_{d}"]})
    fig.legend(handles=[Line2D([], [], color=C_FULL, lw=2, label="best model on Full"),
                        Line2D([], [], color=C_DEMO, lw=2, label="best model on Demo"),
                        Line2D([], [], color=LINE, lw=1.1, label="other models")],
               loc="lower center", ncol=3, frameon=False, fontsize=FS_SMALL, bbox_to_anchor=(0.5, -0.01))
    # centre the "Full/Demo: RSCE, rank 1 at top" headers vertically between the panel title and the rank-1 labels
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, hdr, row1 in headers:
        y_title = ax.title.get_window_extent(rend).y0
        y_row1 = max(t.get_window_extent(rend).y1 for t in row1)
        y_mid = ax.transData.inverted().transform((0.0, (y_title + y_row1) / 2))[1]
        for t in hdr:
            t.set_y(y_mid)
    save(fig, outdir, "Fig3_rank_reproducibility", outputs)


# =========================================================================== FIGURE 4
def fig4(D: Data, outdir: Path, outputs: List[Path], src_rows: List[Dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(W2, 6.9))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.915, bottom=0.085, hspace=0.66, wspace=0.62)
    DEGEN_SPEC = 0.05  # Demo operating points with pooled specificity below this are marked (threshold flags ~all rows)
    specs = [
        ("ppv_std_pooled_mean", "PPV$_{std}$ (folds pooled per repeat; mean of 3 repeats)", "Standardised PPV"),
        ("sens_pooled_mean", "Test sensitivity (threshold set on the training fold)", "Sensitivity"),
        ("spec_pooled_mean", "Test specificity at the same threshold", "Specificity"),
    ]
    k = 0
    for i, (col, xlabel, short) in enumerate(specs):
        for j, d in enumerate(DOMAINS):
            af, ad = D.ppv_agg[(d, "full")], D.ppv_agg[(d, "demo")]
            pc = D.ppv_cmp[d]
            order = list(af.sort_values("ppv_std_pooled_mean", ascending=False).index)
            y = {m: r for r, m in enumerate(order)}
            pi_ref = float(af["pi_ref"].iloc[0])
            tgt = float(af["target_sens"].iloc[0])
            if not (af["target_sens"].eq(tgt).all() and ad["target_sens"].eq(tgt).all() and ad["pi_ref"].eq(pi_ref).all()):
                raise SystemExit(f"STOP: {d}: Full and Demo PPV runs differ in target sensitivity or reference prevalence")
            degen = [m for m in order if float(ad.loc[m, "spec_pooled_mean"]) < DEGEN_SPEC]
            ax = axes[i, j]
            for m in order:
                vf, vd = float(af.loc[m, col]), float(ad.loc[m, col])
                ax.plot([vd, vf], [y[m], y[m]], color=LINE, lw=1.3, zorder=2, solid_capstyle="butt")
                if col == "ppv_std_pooled_mean":
                    for rel, v, lo, hi, cc in (
                            ("Full", vf, pc.loc[m, "full_boot_ci_low"], pc.loc[m, "full_boot_ci_high"], C_FULL),
                            ("Demo", vd, pc.loc[m, "demo_boot_ci_low"], pc.loc[m, "demo_boot_ci_high"], C_DEMO)):
                        ax.plot([lo, hi], [y[m], y[m]], color=cc, lw=3.0, alpha=0.40, zorder=3, solid_capstyle="butt")
                        src_rows.append({"figure": f"Fig4{'ab'[j]}_ci", "domain": d, "model": m, "release": rel,
                                         "metric": col, "value": v, "ci_low": float(lo), "ci_high": float(hi),
                                         "source": INPUTS[f"ppv_cmp_{d}"]})
                ax.scatter([vf], [y[m]], s=30, color=C_FULL, zorder=5, edgecolor="white", lw=0.7)
                if m in degen:
                    ax.scatter([vd], [y[m]], s=26, color=C_DEMO, marker="D", zorder=6, edgecolor=INK, lw=0.8)
                else:
                    ax.scatter([vd], [y[m]], s=30, color=C_DEMO, zorder=5, edgecolor="white", lw=0.7)
                src_rows += [{"figure": f"Fig4{'abcdef'[k]}", "domain": d, "model": m, "release": "Full", "metric": col,
                              "value": vf, "source": INPUTS[f"ppv_agg_{d}_full"]},
                             {"figure": f"Fig4{'abcdef'[k]}", "domain": d, "model": m, "release": "Demo", "metric": col,
                              "value": vd, "source": INPUTS[f"ppv_agg_{d}_demo"],
                              "flag_demo_spec_below_0.05": int(m in degen)}]
            if col == "ppv_std_pooled_mean":
                ax.axvline(pi_ref, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
                ax.text(pi_ref, -0.95, f"  π$_{{ref}}$ = {pi_ref:.4f} (Full prevalence)", ha="left", va="center",
                        fontsize=FS_SMALL, color=INK2)
                xl = (0.0, 0.105) if d == "hosp" else (0.44, 0.70)
            elif col == "sens_pooled_mean":
                ax.axvline(tgt, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
                ax.text(tgt, -0.95, f"  target {tgt:.2f}", ha="left", va="center", fontsize=FS_SMALL, color=INK2)
                xl = (0.6, 1.03)
            else:
                xl = (-0.03, 1.0)
            ax.set_xlim(*xl)
            ax.set_yticks(range(len(order)), [MODEL_LABEL[m] for m in order])
            ax.set_ylim(len(order) - 0.45, -1.45)
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(xlabel, fontsize=FS_SMALL)
            ax.set_title(short, fontsize=FS, loc="left", pad=3, fontweight="bold")
            panel_letter(ax, "abcdef"[k], x=-0.40, y=1.02, ha="left")
            k += 1
    for j, d in enumerate(DOMAINS):
        bb = axes[0, j].get_position()
        fig.text((bb.x0 + bb.x1) / 2 - 0.06, 0.975, DOMAIN_TITLE[d], ha="center", va="center", fontsize=FS + 1,
                 fontweight="bold")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_FULL, label="Full", ms=5),
                        Line2D([], [], marker="o", ls="", color=C_DEMO, label="Demo", ms=5),
                        Line2D([], [], color=LINE, lw=3.0, alpha=0.6, label="95% fold-bootstrap CI (PPV$_{std}$ only)"),
                        Line2D([], [], marker="D", ls="", color=C_DEMO, mec=INK, ms=4.5,
                               label=f"Demo threshold degenerate (specificity < {DEGEN_SPEC:g})")],
               loc="lower center", ncol=4, frameon=False, fontsize=FS_SMALL, bbox_to_anchor=(0.5, -0.003),
               columnspacing=1.2, handletextpad=0.4)
    save(fig, outdir, "Fig4_ppv_std_full_vs_demo", outputs)


# =========================================================================== FIGURE 5
def fig5(D: Data, outdir: Path, outputs: List[Path], src_rows: List[Dict]) -> None:
    c1, c3 = C_COMP["R"], C_COMP["S"]  # top-1 (violet), top-3 (aqua); validated pair, labelled in the legend
    fig = plt.figure(figsize=(W2, 4.1))
    outer = fig.add_gridspec(1, 2, left=0.135, right=0.985, top=0.835, bottom=0.165, wspace=0.42)
    pools = [("all", "All Full patients"), ("icu", "Full patients with an ICU stay")]
    nulls = [("random", "Random"), ("matched", "Prevalence-matched")]
    ypos = [0.0, 1.0]
    n_models = len(D.tw_fullref[("all", "hosp")])
    for j, d in enumerate(DOMAINS):
        fbest = {pl: D.tw_exp3[(pl, d, "random")]["full_best_model"] for pl in TW_DIR}
        if fbest["all"] != fbest["icu"]:
            raise SystemExit("STOP: whole-Full best model differs between the two runs")
        demo_prev = D.tw_runinfo[("all", d)]["null"]["random"]["demo_prevalence"]
        sub = outer[0, j].subgridspec(2, 1, hspace=0.42)
        for i, (pl, pool_lab) in enumerate(pools):
            ax = fig.add_subplot(sub[i])
            ylabels: List[str] = []
            for (n, lab), y in zip(nulls, ypos):
                e = D.tw_exp3[(pl, d, n)]
                p1, l1, h1 = (float(e[c]) for c in ("subsample_P(best_matches_full)", "subsample_P_wilson95_low",
                                                     "subsample_P_wilson95_high"))
                p3, l3, h3 = (float(e[c]) for c in ("subsample_P(full_best_in_top3)",
                                                     "subsample_P(full_best_in_top3)_wilson95_low",
                                                     "subsample_P(full_best_in_top3)_wilson95_high"))
                for yy, p_, lo, hi, col, mk in ((y - 0.17, p1, l1, h1, c1, "o"), (y + 0.17, p3, l3, h3, c3, "D")):
                    ax.plot([lo, hi], [yy, yy], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
                    ax.scatter([p_], [yy], s=30 if mk == "o" else 24, marker=mk, color=col, edgecolor="white", lw=0.7,
                               zorder=3)
                for xx, yy, txt, cc in ((h1 + 0.010, y - 0.17, f"{p1:.3f}", INK), (h3 + 0.010, y + 0.17, f"{p3:.3f}", INK2)):
                    ax.text(xx, yy, txt, va="center", ha="left", fontsize=FS_SMALL, color=cc, zorder=4,
                            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
                prev = D.tw_runinfo[(pl, d)]["null"][{"random": "random", "matched": "prevalence_matched"}[n]]["prevalence_mean"]
                ylabels.append(f"{lab}\nnull prev. {prev:.3f}")
                src_rows.append({"figure": f"Fig5{'ab'[j]}", "domain": d, "pool": pl, "null": n,
                                 "full_best_model": e["full_best_model"], "P_top1": p1, "P_top1_wilson_low": l1,
                                 "P_top1_wilson_high": h1, "P_top3": p3, "P_top3_wilson_low": l3,
                                 "P_top3_wilson_high": h3, "null_prevalence_mean": prev, "demo_prevalence": demo_prev,
                                 "demo_full_best_in_top1": int(e["demo_full_best_in_top1"]),
                                 "demo_full_best_in_top3": int(e["demo_full_best_in_top3"]),
                                 "source": INPUTS[f"tw_{pl}_{d}_{n}_exp3"]})
                if int(e["demo_full_best_in_top3"]) != 0:
                    raise SystemExit("STOP: the Fig. 5 note assumes the real Demo never has the Full-best model in its top 3")
            for kk in (1, 3):
                xv = kk / n_models
                ax.axvline(xv, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
                ax.text(xv + 0.006, 1.47, f"{kk}/{n_models}", ha="left", va="bottom", fontsize=FS_SMALL, color=MUTED)
            ax.set_yticks(ypos, ylabels)
            ax.tick_params(axis="y", labelsize=FS_SMALL)
            ax.set_ylim(1.5, -0.5)
            ax.set_xlim(0, 0.5)
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.set_title(pool_lab, fontsize=FS_SMALL, fontweight="bold", loc="center", pad=3)
            if i == 0:
                ax.text(0.5, 1.17, f"{DOMAIN_TITLE[d]}\nFull-best {MODEL_LABEL[fbest['all']]} · Demo prev. {demo_prev:.3f}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=FS, linespacing=1.15)
                ax.text(-0.36, 1.17, "ab"[j], transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", ha="left",
                        va="bottom")
            else:
                ax.set_xlabel("Probability over 1,000 Demo-sized null draws")
    fig.legend(handles=[Line2D([], [], marker="o", color=c1, lw=1.6, ms=5, label="sample picks the Full-best model"),
                        Line2D([], [], marker="D", color=c3, lw=1.6, ms=4.5, label="Full-best model in the sample's top 3"),
                        Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label=f"uniform random choice among {n_models} models")],
               loc="lower center", ncol=3, frameon=False, fontsize=FS_SMALL, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2)
    save(fig, outdir, "Fig5_trustworthiness_null_full_best_recovery", outputs)

# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="project root (holds Results/)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only", nargs="*", choices=["1", "2", "3", "4", "5"], default=None)
    a = ap.parse_args()
    root, outdir = Path(a.root), Path(a.outdir)
    (outdir / "source_data").mkdir(parents=True, exist_ok=True)
    family = setup_style()
    D = Data(root)
    checks = check_inputs(D)
    print(f"[figures] {len(checks)} input checks passed; font: {family}")
    outputs: List[Path] = []
    todo = a.only or ["1", "2", "3", "4", "5"]
    builders = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}
    for k in todo:
        rows: List[Dict] = []
        builders[k](D, outdir, outputs, rows)
        stem = {"1": "Fig1", "2": "Fig2", "3": "Fig3", "4": "Fig4", "5": "Fig5"}[k]
        p = outdir / "source_data" / f"{stem}_source_data.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        outputs.append(p)
        print(f"[figures] Figure {k} done")
    manifest = {
        "script": "Pipeline/05_figures/make_figures.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "font_family": family,
        "platform": {"python": platform.python_version(), "system": platform.system()},
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {v: sha256(root / v) for v in INPUTS.values()},
        "input_checks_passed": checks,
        "outputs": {str(p.relative_to(outdir)).replace("\\", "/"): sha256(p) for p in outputs},
    }
    (outdir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[figures] wrote {len(outputs)} files + figure_manifest.json to {outdir}")


if __name__ == "__main__":
    main()
