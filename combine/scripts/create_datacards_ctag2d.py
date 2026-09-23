#!/usr/bin/env python3
"""
Create combine histograms and datacards for H+c → ZZ → 4l -- ctag2d COMPARISON VARIANT.

Byte-identical to create_datacards.py except: (1) the c-tag systematics block swaps the
1D fixed-WP CMS_ctag_b/CMS_ctag_c nuisances for the single CMS_ctag2d nuisance (2D
pseudo-continuous SF, analysis/corrections/ctag2d.py) -- this variant is meant to be run
against parquets produced by the hplusc_mva_4class_ctag2d workflow (event_weights.ctagging_2d:
true), NOT the production hplusc_mva_4class workflow, which never writes a weight_CMS_ctag2d_*
column. (2) default --scored-dir/--sumw-dir/--output point at the ctag2d workflow's own
output tree so this never collides with or reads production data. See README_HcZZ.md "Open
items" and second-brain memory hczz_pnet_ctag_sf_migration. Created 2026-08-14; do not promote
by renaming over create_datacards.py -- keep both while comparing.

Canonical copy as of 2026-08-17: this file, under Higgscharmnew/higgscharm/combine/scripts/ (part of
the consolidated ctag2d pipeline -- see ../README.md and ../run_pipeline.sh for the
one-command build+combine flow). The original b-hive_ttcc/combine/create_datacards_ctag2d.py
is now a historical snapshot predating the --merge-bin-ranges/--skip-zx options below;
edit THIS copy going forward.

2026-08-17 additions (see outputs/combine_run3_90160_ctag2d_negbin_rebinning/README.md for
the full investigation these came from):
  --merge-bin-ranges  "surgical rebin" -- merges specific sparse low-score quantile bins
                       into wide bins post-hoc, fixing the negative-bin issue found in
                       Signal's templates at zero measured sensitivity cost. Empirically
                       tuned for the current data; re-diagnose (see
                       outputs/.../scripts/01-05) if the underlying parquets change enough
                       to shift the quantile edges.
  --skip-zx            excludes Z+X entirely (no process/row), writes datacard_no_zx.txt.
  Both default OFF (unchanged original 20-bin, with-ZX behavior) for backward compatibility.
  Recommended production settings, per the 2026-08-16 investigation:
  --merge-bin-ranges "2-4,7-10" --skip-zx  (r=309.5, 0/46 negative-bin histograms).

Original docstring follows unchanged:

5 processes: qqZZ, ggZZ, Signal (H+c), Other_Higgs, ZX (reducible background)
Discriminant: mva_score_Signal (4-class MLP output, P(H+c))
Mass window: [90, 160] GeV

Normalization strategy
  MC: xs × lumi × (window_weight / sumw_generated), where sumw_generated is the
      true pre-selection sum of genWeight for the full processed MC sample
      (read from the sumw/*.json sidecars written by the coffea processor),
      NOT the post-selection weight sum of the scored parquets. Using the
      post-selection sum as the denominator would silently divide out the
      baseline analysis selection efficiency (>=1 ZZ candidate, >=1 c-jet),
      which differs a lot between signal and background.
  ZX: sum of zx_weight from 3P1F + 2P2F parquets (already in mass window)

Using the TEST set only (via test filelist) avoids overtraining bias in shapes.
The scored parquets provide mva_score_Signal; normalization uses xs × lumi scaling.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
import hist
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]

LUMINOSITY_PB = {
    "2022preEE":    7980.4,
    "2022postEE":  26671.7,
    "2023preBPix": 17794.0,
    "2023postBPix": 9450.0,
}
TOTAL_LUMI_PB = sum(LUMINOSITY_PB.values())  # 61896.1 pb^-1

# Cross-sections (pb)
PROCESS_XS_PB = {
    "Signal":      2.6381e-05,
    "ggZZ":        3 * 0.003 + 3 * 0.006,
    "qqZZ":        1.39,
    "Other_Higgs": (
        0.01434 + 0.00112 + 0.000775 + 0.000156 + 0.000244
        + 0.00312 + 0.000144 + 0.000173
    ),
}
# gg->ZZ is generated at LO (MCFM701) -- the physical yield needs the
# NNLO/LO k-factor (HNNLO-computed, ~2.27 at m(ZZ)=125 GeV, CMS H->ZZ->4l
# convention). The "kfactor_ggZZ" 1.10 lnN below is the UNCERTAINTY on this
# k-factor, not the k-factor itself -- propagated from the fix + full
# investigation in create_datacards_ctag2d_100150.py (2026-09-14, see
# hczz_physicsdays_systematics_audit memory); this script isn't the live
# production path but kept consistent with it.
GGZZ_KFACTOR = 2.27

EXPECTED_YIELDS = {p: xs * TOTAL_LUMI_PB for p, xs in PROCESS_XS_PB.items()}
EXPECTED_YIELDS["ggZZ"] *= GGZZ_KFACTOR

CLASS_NAMES = ["qqZZ", "ggZZ", "Signal", "Other_Higgs"]

# Sample name (substring) → process name
PROCESS_MAPPING = {
    "SomeSMSignal":                 "Signal",
    "HPlusCharm":                   "Signal",
    "ZZto4L":                       "qqZZ",
    "GluGlutoContinto2Zto4E":       "ggZZ",
    "GluGlutoContinto2Zto4Mu":      "ggZZ",
    "GluGlutoContinto2Zto4Tau":     "ggZZ",
    "GluGluToContinto2Zto2E2Mu":    "ggZZ",
    "GluGluToContinto2Zto2E2Tau":   "ggZZ",
    "GluGluToContinto2Zto2Mu2Tau":  "ggZZ",
    "GluGluHtoZZto4L":              "Other_Higgs",
    "VBFHto2Zto4L":                 "Other_Higgs",
    "WminusH_Hto2Zto4L":           "Other_Higgs",
    "WplusH_Hto2Zto4L":            "Other_Higgs",
    "ZHto2Zto4L":                   "Other_Higgs",
    "TTH_Hto2Z":                    "Other_Higgs",
    "bbH_Hto2Zto4L":               "Other_Higgs",
    "HPlusBottom":                  "Other_Higgs",
    "2022preEEHB":                  "Other_Higgs",
    "2022postEEHB":                 "Other_Higgs",
    "2023preBPixHB":                "Other_Higgs",
    "2023postBPixHB":               "Other_Higgs",
}

MASS_WINDOW = (90, 160)
# Placeholder default; main() overwrites this at runtime (equal-width or
# qqZZ-weighted quantile bins, see --binning / compute_quantile_bins()).
MVA_BINS = np.linspace(0, 1, 21)  # 20 bins

# ---------------------------------------------------------------------------
# Shape systematics from per-event weight variations already in the scored
# parquets. "col" is the base weight column name (Up/Down appended); era_dep
# means the column name is suffixed by year (2022 vs 2023).
# ---------------------------------------------------------------------------
SYSTEMATICS = {
    "lhe_pdf":        {"col": "weight_lhe_pdf",       "era_dep": False},
    "lhe_alphaS":     {"col": "weight_lhe_alphaS",    "era_dep": False},
    "scalevar_muR":   {"col": "weight_scalevar_muR",  "era_dep": False},
    "scalevar_muF":   {"col": "weight_scalevar_muF",  "era_dep": False},
    "ps_isr":         {"col": "weight_ps_isr",        "era_dep": False},
    "ps_fsr":         {"col": "weight_ps_fsr",        "era_dep": False},
    "CMS_pileup":     {"col": "weight_CMS_pileup",    "era_dep": True},
    # ctag2d variant: the 2D pseudo-continuous SF (analysis/corrections/ctag2d.py) is ONE
    # per-jet, per-event weight already covering b/c/light flavors together (unlike the
    # production 1D scheme's three separate CMS_ctag_b/CMS_ctag_c/CMS_ctag_light nuisances,
    # the last of which is excluded there for the eff/(1-eff) sign-flip bug -- see
    # create_datacards.py). It has no analogous light-flavor sign-flip failure mode (no
    # eff-based tagged/untagged split at all), so there is nothing to exclude here.
    "CMS_ctag2d":     {"col": "weight_CMS_ctag2d",    "era_dep": True},
    # Electron reco efficiency, split by pT range -- already computed and stored
    # (weight_CMS_eff_e_reco_<range>_<year>Up/Down) but never wired in before
    # 2026-08-17. Verified sane for every process incl. Signal/HPlusBottom
    # (<=6% shift, properly symmetric +/-) -- detector-level correction, not
    # LHE-generator weight, so unaffected by the private-sample LHE bugs above.
    "CMS_eff_e_reco_20to75":  {"col": "weight_CMS_eff_e_reco_20to75",  "era_dep": True},
    "CMS_eff_e_reco_above75": {"col": "weight_CMS_eff_e_reco_above75", "era_dep": True},
    "CMS_eff_e_reco_below20": {"col": "weight_CMS_eff_e_reco_below20", "era_dep": True},
    # Muon ID efficiency SF (loose WP, see hplusc_mva_4class_ctag2d.yaml's
    # `muon:` block, wired 2026-08-21). Detector-level correction like
    # CMS_eff_e_reco above, not an LHE-generator weight -- unaffected by the
    # private-sample lhe_pdf/scalevar bugs, usable for every process.
    "CMS_eff_m_id":           {"col": "weight_CMS_eff_m_id",           "era_dep": True},
}

# Per-process usable systematics, based on empirical sanity checks (2026-08-03/04):
# Signal's lhe_pdf/lhe_alphaS/scalevar_muR/scalevar_muF are anomalous.
# lhe_pdf: root cause confirmed (missing 1/sqrt(N-1) NNPDF-replica
# normalization in analysis/corrections/lhepdf.py, now fixed there for
# future reprocessing) -- but an algebraic downstream correction of the
# EXISTING parquets' weight_lhe_pdfUp/Down still left Signal's PDF
# uncertainty at ~88% (vs qqZZ's 4.3%, Other_Higgs's 1.8%), well outside any
# sane range. Rather than retrofit an unverified correction onto data not
# independently confirmed to need exactly this factor, Signal's lhe_pdf
# stays excluded until a genuine reprocessing with the fixed source code can
# be validated. lhe_alphaS is separately degenerate (Signal's PDF set has
# only 101 members, no alphaS variation at all). scalevar_muR/muF show a
# DIFFERENT, still-undiagnosed bug (Up and Down both shifted the same
# direction relative to nominal, unlike the symmetric +/-delta pattern seen
# for qqZZ/Other_Higgs) -- likely specific to how this private sample's
# LHEScaleWeight branch was filled. CMS_ctag_b was degenerate for Signal
# under the OLD 1D scheme (Up==nominal) -- NOT re-verified for CMS_ctag2d,
# since that observation was specific to the 1D corrector's per-flavor SF
# lookup; re-check once real ctag2d-reprocessed Signal parquets exist before
# trusting this systematic for Signal. ggZZ has no LHEScaleWeight branch at
# all and degenerate lhe_pdf/lhe_alphaS.
#
# Other_Higgs is a pooled process that includes HPlusBottom (private H+b,
# same LHE production chain as Signal/HPlusCharm) alongside official
# centrally-produced samples (ggH/VBF/WH/ZH/ttH/bbH to 4l). Decomposed by
# sub-sample (2026-08-04, [115,135] window): the official samples are sane
# (lhe_pdf +2.8%/-2.8%, scalevar_muR +16.4%/-12.5%, properly antisymmetric),
# but HPlusBottom alone shows the *identical* two bugs already diagnosed for
# Signal -- lhe_pdf blown up and asymmetric (+857%/-657%) and scalevar_muR/muF
# with Up and Down both shifted the same direction (+70.7%/+73.7% and
# +51.2%/+85.4%). HPlusBottom is only ~4% of Other_Higgs's nominal yield but
# dominates the shape variation, which is what caused Combine's "Bogus norm"
# crash on Other_Higgs's lhe_pdf Down once the per-event ratio clip (that had
# been masking this) was removed. Excluding lhe_pdf/scalevar_muR/scalevar_muF
# from Other_Higgs here, matching the treatment already given to Signal for
# the same underlying private-sample production bug. lhe_alphaS and
# CMS_ctag_b are larger for HPlusBottom than the official samples but not
# pathological (no sign flip, no same-direction Up/Down) -- kept.
# lhe_alphaS EXCLUDED EVERYWHERE, 2026-08-19 (under investigation, not yet
# fixed): analysis/corrections/lhepdf.py's delta_alpha = 0.5 * np.abs(w_as_high
# - w_as_low) takes an absolute value before symmetrizing, which forces
# w_up_alpha = 1 + delta_alpha >= 1 and w_down_alpha = 1 - delta_alpha <= 1 for
# EVERY event of EVERY process unconditionally -- the true sign of the alphaS
# variation is destroyed, so "Up" is mathematically guaranteed to raise every
# process's yield and "Down" to lower it, regardless of actual physics. This
# is why qqZZ and Other_Higgs were seen moving in lockstep (rank #3 impact,
# 17.6) -- not a real correlated physics effect, an artifact of the abs().
# lhe_pdf's use of a symmetric envelope (sqrt(sum of squared diffs)) is
# correct for its 100-eigenvector PDF4LHC convention; alphaS has only 2
# members and needs a genuinely signed shift instead -- different case, same
# function, only alphaS is broken. See README_HcZZ.md's 2026-08-19 dated
# update under Systematics for the full writeup. Re-include once
# lhepdf.py's delta_alpha is fixed (drop the np.abs()) and re-verified.
EFF_E_RECO = {"CMS_eff_e_reco_20to75", "CMS_eff_e_reco_above75", "CMS_eff_e_reco_below20"}
USABLE_SYST = {
    "qqZZ":        set(SYSTEMATICS.keys()) - {"lhe_alphaS"},
    "Other_Higgs": set(SYSTEMATICS.keys()) - {"lhe_pdf", "scalevar_muR", "scalevar_muF", "lhe_alphaS"},
    "ggZZ":        {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"} | EFF_E_RECO,
    "Signal":      {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"} | EFF_E_RECO,
}


def era_year(era):
    return "2022" if era.startswith("2022") else "2023"


def syst_col(syst_name, direction, era):
    """direction: 'Up' or 'Down'. Returns the parquet column name for this
    systematic/direction/era."""
    spec = SYSTEMATICS[syst_name]
    base = spec["col"]
    if spec["era_dep"]:
        base = f"{base}_{era_year(era)}"
    return f"{base}{direction}"


def get_process_from_path(filepath):
    # Match longest key first: "ZZto4L" is a literal substring of
    # "GluGluHtoZZto4L" (a different, Other_Higgs-class process), so naive
    # dict-insertion-order matching silently misrouted every GluGluHtoZZto4L
    # event into qqZZ for the entire session (found 2026-08-04 via a
    # mean-mva_score_Signal per-subdirectory diagnostic: GluGluHtoZZto4L
    # scored ~0.29 vs true ZZto4L's ~0.04, dragging the whole qqZZ AUC down).
    for key in sorted(PROCESS_MAPPING, key=len, reverse=True):
        if key in filepath:
            return PROCESS_MAPPING[key]
    return None


# ---------------------------------------------------------------------------
# Load true pre-selection sumw (from base.py's sumw/*.json sidecars)
# ---------------------------------------------------------------------------

def load_sumw(sumw_dir):
    """
    Walk <sumw_dir>/<era>/<dataset_partition>/sumw/*.json and aggregate the
    pre-selection sum of genWeight per process (summed over all partitions
    and eras). This is the true full-generated-sample denominator, as opposed
    to the post-selection weight sum available in the scored parquets.
    """
    import json

    sumw_by_proc = defaultdict(float)
    n_files = defaultdict(int)

    for era in ERAS:
        era_dir = os.path.join(sumw_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: sumw era dir not found: {era_dir}")
            continue

        for dataset_partition in os.listdir(era_dir):
            proc = get_process_from_path(dataset_partition)
            if proc is None:
                continue

            sumw_glob = os.path.join(era_dir, dataset_partition, "sumw", "*.json")
            for jf in glob.glob(sumw_glob):
                try:
                    with open(jf) as f:
                        rec = json.load(f)
                except Exception as e:
                    print(f"  WARNING: {jf}: {e}")
                    continue
                sumw_by_proc[proc] += rec["sumw"]
                n_files[proc] += 1

    print(f"\n  {'Process':<15} {'n_sumw_files':>14} {'sumw_generated':>16}")
    print("  " + "-" * 47)
    for proc in CLASS_NAMES:
        print(f"  {proc:<15} {n_files.get(proc, 0):>14} {sumw_by_proc.get(proc, 0.0):>16.2f}")

    return sumw_by_proc


# ---------------------------------------------------------------------------
# Load MC from test-set scored parquets
# ---------------------------------------------------------------------------

def load_mc_scored_parquets(scored_dir, sumw_by_proc):
    """
    Walk all eras/samples in scored_dir, read mva_score_Signal + weight_nominal.
    Normalization: scale weight_nominal by (xs × lumi) / sum(all raw weights per process),
    so the histogram integral equals the expected physics yield in the mass window.

    Returns dict: proc → (scores array, scaled_weights array)
    """
    # Two-pass: first collect all data, then apply xs scaling
    raw_weight  = defaultdict(float)
    proc_scores = defaultdict(list)
    proc_weights = defaultdict(list)
    # proc_syst_weights[proc][syst][direction] -> list of arrays, same event
    # order/mask as proc_scores/proc_weights so they align 1:1
    proc_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for era in ERAS:
        era_dir = os.path.join(scored_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue

        for sample_name in os.listdir(era_dir):
            proc = get_process_from_path(sample_name)
            if proc is None:
                continue

            sample_dir = os.path.join(era_dir, sample_name, "base")
            if not os.path.isdir(sample_dir):
                continue

            usable = USABLE_SYST.get(proc, set())
            syst_cols_wanted = []
            for syst in usable:
                for direction in ("Up", "Down"):
                    syst_cols_wanted.append(syst_col(syst, direction, era))

            for pq_file in glob.glob(os.path.join(sample_dir, "*.parquet")):
                base_cols = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]
                try:
                    df = pq.read_table(pq_file, columns=base_cols + syst_cols_wanted).to_pandas()
                except Exception:
                    # Some requested systematic columns may not exist in this
                    # file/sample -- fall back to reading only what's present.
                    try:
                        available = pq.ParquetFile(pq_file).schema.names
                        cols = base_cols + [c for c in syst_cols_wanted if c in available]
                        df = pq.read_table(pq_file, columns=cols).to_pandas()
                    except Exception as e:
                        print(f"  WARNING: {pq_file}: {e}")
                        continue

                w_all = df["weight_nominal"].fillna(0).values
                raw_weight[proc] += w_all.sum()

                # Apply mass window
                mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & \
                       (df["zz_mass_inclusive"] < MASS_WINDOW[1])
                df_win = df[mask]
                valid = df_win["mva_score_Signal"] >= 0
                df_win = df_win[valid]
                if len(df_win) == 0:
                    continue

                proc_scores[proc].append(df_win["mva_score_Signal"].values)
                nominal_win = df_win["weight_nominal"].fillna(0).values
                proc_weights[proc].append(nominal_win)

                for syst in usable:
                    for direction in ("Up", "Down"):
                        col = syst_col(syst, direction, era)
                        if col in df_win.columns:
                            proc_syst_weights[proc][syst][direction].append(
                                df_win[col].fillna(0).values
                            )
                        else:
                            # Column missing for this file -- fall back to
                            # nominal (no variation) so array lengths align.
                            proc_syst_weights[proc][syst][direction].append(nominal_win)

    # Apply xs × lumi normalization using the TRUE pre-selection sumw
    # (sumw_by_proc, from base.py's sumw/*.json sidecars) as the denominator.
    # `raw` (post-baseline-selection weight sum, from the scored parquets
    # themselves) is kept only for a diagnostic comparison against sumw: the
    # ratio raw/sumw is the baseline analysis selection efficiency that was
    # previously being silently divided out of the normalization.
    mc_data = {}
    print(f"\n  {'Process':<15} {'raw_wgt':>12} {'sumw_gen':>14} {'baseline_eff':>13} {'expected':>10} {'in_win_yield':>14}")
    print("  " + "-" * 82)

    for proc in CLASS_NAMES:
        raw = raw_weight.get(proc, 0.0)
        sumw_gen = sumw_by_proc.get(proc, 0.0)
        exp = EXPECTED_YIELDS.get(proc, 0.0)
        baseline_eff = raw / sumw_gen if sumw_gen > 0 else float("nan")

        scores  = np.concatenate(proc_scores[proc])  if proc_scores[proc]  else np.array([])
        weights = np.concatenate(proc_weights[proc]) if proc_weights[proc] else np.array([])

        if sumw_gen > 0 and exp > 0:
            scale = exp / sumw_gen
            scaled_weights = weights * scale
        else:
            print(f"  WARNING: no sumw for {proc} -- falling back to post-selection raw_weight")
            scale = exp / raw if raw > 0 and exp > 0 else 1.0
            scaled_weights = weights * scale

        # Same scale factor applied to each systematic-varied weight column --
        # the variation is already a full replacement weight, so this yields
        # the correctly-scaled Up/Down expected yield in the mass window.
        # No ratio clipping: an out-of-range per-event weight variation is a
        # symptom of a real bug upstream (see CMS_ctag_light above), not
        # numerical noise to paper over -- if it recurs, the fix belongs in
        # the corrections code, not here.
        syst_scaled = defaultdict(dict)
        for syst, by_dir in proc_syst_weights[proc].items():
            for direction, arrs in by_dir.items():
                w = np.concatenate(arrs) if arrs else np.array([])
                syst_scaled[syst][direction] = w * scale

        mc_data[proc] = (scores, scaled_weights, syst_scaled)
        in_win = scaled_weights.sum()
        print(f"  {proc:<15} {raw:>12.2f} {sumw_gen:>14.2f} {baseline_eff:>13.4%} {exp:>10.4f} {in_win:>14.4f}")

    return mc_data


# ---------------------------------------------------------------------------
# Load Z+X
# ---------------------------------------------------------------------------

def load_zx_parquets(zx_dir):
    """
    Load all eras, 3P1F + 2P2F. Events already in mass window.
    Returns (scores array, weights array).
    """
    all_scores, all_weights = [], []
    for era in ERAS:
        for cr in ("3p1f", "2p2f"):
            fp = os.path.join(zx_dir, era, f"zx_{cr}_{era}_mva.parquet")
            if not os.path.exists(fp):
                print(f"  WARNING: missing {fp}")
                continue
            df = pd.read_parquet(fp)
            all_scores.append(df["mva_score_Signal"].values)
            all_weights.append(df["zx_weight"].values)

    if not all_scores:
        return np.array([]), np.array([])

    scores  = np.concatenate(all_scores)
    weights = np.concatenate(all_weights)
    print(f"  ZX (all eras)  n_events={len(scores)}  net_yield={weights.sum():.2f}")
    return scores, weights


# ---------------------------------------------------------------------------
# Variable-width ("quantile") binning
# ---------------------------------------------------------------------------

def compute_quantile_bins(mc_data, n_bins=20, reference_proc="qqZZ"):
    """
    Equal-width [0,1] bins waste most of their resolution on mva_score_Signal:
    the score distribution is heavily skewed toward 0 for both signal and
    background (found 2026-08-04: 71.8% of qqZZ and 41.4% of Signal fall in
    the single first equal-width bin, leaving 14/20 bins with <1% of events
    each). This crushes most of the shape information Combine can exploit
    into one bin, even though the classifier's raw separation power (AUC
    ~0.66, on par with the single-variable leadingjet_cvsl tagger at ~0.67)
    is fine -- it's a binning problem, not a classifier problem.

    Build bin edges as weighted quantiles of `reference_proc`'s score
    distribution (qqZZ, since it's ~91% of background and this is the
    standard "roughly-equal background yield per bin" convention) so
    resolution concentrates where the dominant background actually has
    statistics, rather than being wasted on empty equal-width bins.
    """
    scores, weights, _ = mc_data[reference_proc]
    w = np.abs(weights)  # rank by magnitude; sign doesn't affect quantile position
    order = np.argsort(scores)
    scores_sorted, w_sorted = scores[order], w[order]
    cumw = np.cumsum(w_sorted)
    cumw /= cumw[-1]

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.interp(quantiles, cumw, scores_sorted)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)  # dedupe if a point-mass spike collides multiple quantiles

    if len(edges) - 1 < n_bins:
        print(f"  NOTE: requested {n_bins} quantile bins on '{reference_proc}', "
              f"got {len(edges)-1} after deduping degenerate edges (point-mass spike).")
    return edges


# ---------------------------------------------------------------------------
# Surgical rebinning -- merge specific sparse bins post-hoc (2026-08-17)
#
# Found 2026-08-16: Signal's private HPlusCharm/SomeSMSignal LHE sample carries a
# ~26% negative-weight event fraction. In the score region where Signal has almost
# no statistics (qqZZ-quantile binning concentrates resolution where qqZZ lives,
# not Signal), a handful of raw events per bin lets that fraction fluctuate hard
# enough to flip individual bins negative -- 9 of 46 histograms (Signal nominal +
# 8 systematic variants) on the original 20-bin scheme. Never crashed combine
# (process totals stay positive) but a real, undiagnosed negative bin all the same.
#
# Fix: merge only the specific sparse bins that go negative into wide bins, leaving
# every other bin at full original resolution -- unlike uniform coarsening (10 or 8
# bins), this costs zero measured sensitivity (r unchanged at 309.5) because both
# merged regions were places Signal had essentially no statistics regardless of
# sign. Full diagnosis, event-by-event tracing, and the uniform-coarsening
# comparison that ruled it out: outputs/combine_run3_90160_ctag2d_negbin_rebinning/
# README.md and its scripts/01-05.
# ---------------------------------------------------------------------------

def parse_merge_ranges(spec):
    """Parse '2-4,7-10' into [(2, 4), (7, 10)] -- inclusive, 0-based bin-index
    ranges referring to bins in the just-computed quantile edges array."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-")
        lo, hi = int(lo), int(hi)
        if lo > hi:
            raise ValueError(f"--merge-bin-ranges: invalid range '{part}' (lo > hi)")
        ranges.append((lo, hi))
    return ranges


def merge_bin_edges(edges, merge_ranges):
    """Given quantile bin edges (n_bins+1 values) and a list of inclusive
    bin-index ranges to merge, drop the interior edges inside each range so
    each range becomes one wide bin. Ranges must not overlap."""
    n_bins = len(edges) - 1
    drop_idx = set()
    for lo, hi in merge_ranges:
        if not (0 <= lo <= hi <= n_bins - 1):
            raise ValueError(
                f"--merge-bin-ranges: range {lo}-{hi} out of bounds for "
                f"{n_bins} bins (valid bin indices: 0-{n_bins - 1})"
            )
        for i in range(lo + 1, hi + 1):
            if i in drop_idx:
                raise ValueError(
                    f"--merge-bin-ranges: overlapping ranges detected at edge index {i}"
                )
            drop_idx.add(i)
    return np.array([e for i, e in enumerate(edges) if i not in drop_idx])


# ---------------------------------------------------------------------------
# Build histograms
# ---------------------------------------------------------------------------

def build_histograms(mc_data, zx_scores, zx_weights):
    """
    Returns (histograms, histograms_var): parallel dicts, same keys.
    `histograms[name]` is the per-bin sum-of-weights (as before).
    `histograms_var[name]` is the per-bin sum-of-weights-SQUARED -- the true
    statistical variance of a weighted-MC bin. Needed so write_root_file can
    populate each template's real TH1::fSumw2 (see its docstring for why this
    matters for autoMCStats). Mirrors the fix in create_datacards_ctag2d_100150.py.
    """
    histograms = {}
    histograms_var = {}
    n_bins = len(MVA_BINS) - 1

    print(f"\n  {'Process':<15} {'n_events':>10} {'hist_integral':>14}")
    print("  " + "-" * 42)

    for proc in CLASS_NAMES:
        scores, weights, syst_scaled = mc_data[proc]
        if len(scores) == 0:
            histograms[proc] = np.zeros(n_bins)
            histograms_var[proc] = np.zeros(n_bins)
            print(f"  {proc:<15} {'(empty)':>10}")
            continue
        h, _ = np.histogram(scores, bins=MVA_BINS, weights=weights)
        h2, _ = np.histogram(scores, bins=MVA_BINS, weights=weights ** 2)
        histograms[proc] = h.astype(np.float64)
        histograms_var[proc] = h2.astype(np.float64)
        print(f"  {proc:<15} {len(scores):>10} {h.sum():>14.4f}")

        for syst, by_dir in syst_scaled.items():
            for direction, w in by_dir.items():
                if len(w) != len(scores):
                    print(f"  WARNING: {proc}/{syst}{direction} length mismatch "
                          f"({len(w)} vs {len(scores)}) -- skipping")
                    continue
                hs, _ = np.histogram(scores, bins=MVA_BINS, weights=w)
                hs2, _ = np.histogram(scores, bins=MVA_BINS, weights=w ** 2)
                histograms[f"{proc}_{syst}{direction}"] = hs.astype(np.float64)
                histograms_var[f"{proc}_{syst}{direction}"] = hs2.astype(np.float64)

    if len(zx_scores) > 0:
        h, _ = np.histogram(zx_scores, bins=MVA_BINS, weights=zx_weights)
        h2, _ = np.histogram(zx_scores, bins=MVA_BINS, weights=zx_weights ** 2)
        histograms["ZX"] = h.astype(np.float64)
        histograms_var["ZX"] = h2.astype(np.float64)
        print(f"  {'ZX':<15} {len(zx_scores):>10} {h.sum():>14.4f}")
    else:
        histograms["ZX"] = np.zeros(n_bins)
        histograms_var["ZX"] = np.zeros(n_bins)

    return histograms, histograms_var


# ---------------------------------------------------------------------------
# Write ROOT file
# ---------------------------------------------------------------------------

def write_root_file(histograms, histograms_var, output_dir,
                     filename="histograms_mva_with_zx.root", include_zx=True):
    """
    Each MC/ZX template is written as a Weight()-storage `hist.Hist` so its true
    per-bin variance (sum of w^2, from build_histograms) lands in the ROOT
    file's TH1::fSumw2 array, instead of the previous bare-(values,edges)-tuple
    write (which leaves fSumw2 empty and forces a Poisson-proxy fallback on
    readback -- exactly zero variance for negative-content bins, an
    underestimate everywhere else). This is what was silently starving
    autoMCStats of real per-bin statistics. See
    create_datacards_ctag2d_100150.py's write_root_file docstring for the full
    investigation (found + fixed 2026-09-17, confirmed converges in ~24s where
    the old files failed to converge at all after 11.5h).
    """
    path = os.path.join(output_dir, filename)
    nominal_procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"] + (["ZX"] if include_zx else [])
    with uproot.recreate(path, compression=None) as f:
        for proc, h in histograms.items():
            if proc == "ZX" and not include_zx:
                continue  # --skip-zx: never write the ZX histogram at all
            hw = hist.Hist(hist.axis.Variable(MVA_BINS), storage=hist.storage.Weight())
            hw.view().value = h
            hw.view().variance = histograms_var[proc]
            f[f"h_{proc}"] = hw
        # Asimov data_obs = sum of NOMINAL process histograms only -- must not
        # include the systematic-varied Up/Down histograms also stored here.
        # Left as a plain tuple deliberately -- Combine's binned likelihood
        # treats observed data as Poisson-counted by construction, it doesn't
        # read Sumw2 off data_obs.
        data_obs = sum(histograms[p] for p in nominal_procs if p in histograms)
        f["h_data_obs"] = (data_obs, MVA_BINS)
    print(f"\n  ROOT file: {path}")
    return path


# ---------------------------------------------------------------------------
# Write datacard
# ---------------------------------------------------------------------------

def write_datacard(histograms, root_filename, output_dir, include_zx=True, automcstats_threshold=None):
    dc_path = os.path.join(output_dir, "datacard_mva_with_zx.txt" if include_zx else "datacard_no_zx.txt")

    procs    = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"] + (["ZX"] if include_zx else [])
    proc_idx = list(range(len(procs)))
    yields   = {p: histograms[p].sum() for p in procs}

    col_w = 16

    def row(label, values):
        return label + "".join(f"{v:<{col_w}}" for v in values) + "\n"

    with open(dc_path, "w") as dc:
        dc.write(f"# H+c → ZZ → 4l: MVA shape analysis "
                 f"{'with Z+X background' if include_zx else '-- NO Z+X (MC-only, --skip-zx)'}\n")
        dc.write(f"# Mass window: {MASS_WINDOW[0]}–{MASS_WINDOW[1]} GeV  |  Lumi: {TOTAL_LUMI_PB:.1f} pb^-1\n")
        dc.write(f"# Discriminant: mva_score_Signal (4-class MLP)\n")
        dc.write(f"imax 1\njmax {len(procs)-1}\nkmax *\n")
        dc.write("-" * 80 + "\n")
        dc.write(f"shapes * * {root_filename} h_$PROCESS h_$PROCESS_$SYSTEMATIC\n")
        dc.write("-" * 80 + "\n")
        dc.write("bin         hczz\n")
        dc.write("observation -1\n")
        dc.write("-" * 80 + "\n")
        dc.write(row("bin         ", ["hczz"] * len(procs)))
        dc.write(row("process     ", procs))
        dc.write(row("process     ", [str(i) for i in proc_idx]))
        dc.write(row("rate        ", [f"{yields[p]:.6f}" for p in procs]))
        dc.write("-" * 80 + "\n")

        def syst_row(name, stype, vals):
            line = f"{name:<25}{stype:<8}"
            for p in procs:
                line += f"{vals.get(p, '-'):<{col_w}}"
            return line + "\n"

        dc.write(syst_row("lumi_Run3",     "lnN", {p: "1.014" for p in procs}))
        # QCDscale_gg / pdf_gg / kfactor_ggZZ: rate lnN's retained only where a
        # real per-event shape systematic isn't available (Signal's LHE
        # weights are anomalous -- see analysis/corrections/lhepdf.py note;
        # ggZZ has no LHEScaleWeight branch at all). Values matched to
        # HIG-24-013 (AN2023_157_v10) Table 15 2026-08-28: QCD scale (gg)
        # 3.9%, PDF set (gg) 3.2%, gg->ZZ k-factor 10% -- three separate
        # nuisances in the official analysis, previously collapsed into one
        # 5% pdf_gg placeholder plus a mislabeled "QCDscale_ggZZ" (that 10%
        # value was already correct, just named after the wrong physics --
        # it's the k-factor uncertainty, not a QCD-scale uncertainty).
        # kfactor_ggZZ is the UNCERTAINTY on the GGZZ_KFACTOR=2.27
        # central-value correction now applied above (2026-09-14 fix).
        dc.write(syst_row("QCDscale_gg",   "lnN", {"ggZZ": "1.039", "Signal": "1.039"}))
        dc.write(syst_row("pdf_gg",        "lnN", {"ggZZ": "1.032", "Signal": "1.032"}))
        dc.write(syst_row("kfactor_ggZZ",  "lnN", {"ggZZ": "1.10"}))
        # Added 2026-08-17, sourced from Felix Heyen thesis Appendix D (no
        # per-event weight column exists for either -- pure rate lnN, same
        # treatment as pdf_gg/QCDscale_ggZZ above):
        #   BR_HZZ4l: 2% branching-ratio uncertainty on B(H->ZZ->4l),
        #     applies to every Higgs-produced process (Signal + Other_Higgs).
        #   QCDscale_qqZZ: 4% K-factor uncertainty on qq->ZZ (thesis's
        #     "qqZZkFactor"), alongside the existing ggZZ-only QCDscale_ggZZ
        #     row -- the two K-factors are separate uncertainty sources, not
        #     duplicates of each other.
        # BR_HZZ4l=2% CHASED 2026-09-14, inconclusive-but-plausible (see
        # create_datacards_ctag2d_100150.py for the full note): real
        # HIG-24-013 (arXiv:2501.14849) doesn't apply this at all ("effect of
        # theoretical uncertainties on the signal is negligible... not
        # included in the fit"); primary LHCHWG source (arXiv:1107.5909)
        # gives H->ZZ total uncertainty "at the level of 1%... below 3%", a
        # range not a point value. 2% is plausible, not independently
        # pinned down. No numeric change.
        dc.write(syst_row("BR_HZZ4l",      "lnN", {"Signal": "1.02", "Other_Higgs": "1.02"}))
        # QCDscale_qqZZ=4% CONFIRMED EXACT 2026-09-14 against the real HIG-24-013
        # text (arXiv:2501.14849): "this yields an overall 4% effect" for the
        # qq->ZZ renorm/fact scale variation. High confidence, not a placeholder.
        dc.write(syst_row("QCDscale_qqZZ", "lnN", {"qqZZ": "1.04"}))
        # NOTE (checked 2026-08-21, no code change): HIG-24-013 (AN2023_157_v10,
        # Table 15) quotes a separate "PDF set (qq->ZZ): +3.1/-3.4%" rate
        # uncertainty alongside pdf_gg above -- but unlike ggZZ/Signal (whose
        # lhe_pdf is excluded from USABLE_SYST, hence the pdf_gg lnN
        # placeholder), qqZZ DOES have a real, validated per-event lhe_pdf
        # shape systematic already wired in below (USABLE_SYST["qqZZ"]
        # includes lhe_pdf, measured ~2.8% -- see hczz_combine_systematics
        # memory). Adding a pdf_qq lnN on top would double-count the same
        # PDF uncertainty via two different nuisances. No gap here.
        if include_zx:
            # ZX_norm=30% -- CITED 2026-09-14 (see create_datacards_ctag2d_100150.py
            # for the full investigation): real HIG-24-013 text (arXiv:2501.14849)
            # gives 25-46% depending on final state (CR stats + misID +/-1sigma +
            # composition, quadrature-combined); this single-bin card doesn't split
            # final states, so 30% is a representative point inside that range, not
            # a precisely derived number. Independently cross-checked the CR-stats
            # component alone from pkatris's real production data: 5.61% (all 4
            # eras) -- much smaller than 30%, consistent with the real uncertainty
            # being dominated by misID/composition systematics, not statistics.
            dc.write(syst_row("ZX_norm", "lnN", {"ZX": "1.30"}))

        dc.write(f"\n# Shape systematics from per-event weight variations (Up/Down\n")
        dc.write(f"# histograms in the ROOT file); coefficient 1 = process has it.\n")
        for syst in SYSTEMATICS:
            vals = {p: "1" for p in procs if syst in USABLE_SYST.get(p, set())}
            if not vals:
                continue
            dc.write(syst_row(syst, "shape", vals))

        if include_zx:
            dc.write(f"\n# ZX normalization floats freely in the fit (data-driven)\n")
            dc.write(f"ZX_rate  rateParam  hczz  ZX  1.0  [0.1,10.0]\n")

        # autoMCStats, added 2026-08-18 (threshold 10, matching standard combine
        # convention). CAVEAT, checked this session: several bins in the current
        # quantile binning have MC-stat errors 10-90x the bin content -- this is
        # numerically degenerate with such sparse bins (Barlow-Beeston-lite will
        # add a nuisance per affected bin, not just "cheap" as usually assumed).
        #
        # OFF BY DEFAULT since 2026-08-19 (was silently on-by-default 2026-08-18/19,
        # a real regression -- caught and fixed same day). The --automcstats-threshold
        # sweep (10/50/100) this flag was built to test is done: NONE of the three
        # converge (11.5h each, zero quantiles produced) -- raising the threshold
        # does not help. This is NOT a resolved gap -- per-bin MC-stat uncertainty
        # currently has no coverage at all; see README_HcZZ.md's 2026-08-19 dated
        # update under Systematics for the open follow-up items. Pass
        # --automcstats-threshold <n> explicitly to opt back in for testing.
        if automcstats_threshold is not None:
            dc.write(f"\nhczz autoMCStats {automcstats_threshold}\n")

    print(f"  Datacard:  {dc_path}")
    return dc_path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(histograms, include_zx=True):
    procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"] + (["ZX"] if include_zx else [])
    print(f"\n{'Process':<15} {'Yield in [{},{}] GeV'.format(*MASS_WINDOW):>20}")
    print("-" * 38)
    total_bkg = 0
    for p in procs:
        y = histograms[p].sum()
        if p != "Signal":
            total_bkg += y
        print(f"  {p:<13} {y:>12.4f}")
    print("-" * 38)
    print(f"  {'Total bkg':<13} {total_bkg:>12.4f}")
    sig = histograms["Signal"].sum()
    if total_bkg > 0:
        print(f"  S/sqrt(B) = {sig/total_bkg**0.5:.4f}")


# ---------------------------------------------------------------------------
# Negative-bin scan -- automatic post-build safety net (2026-08-17). Never
# blocks/clips anything (per this project's standing "root-cause, don't paper
# over" rule) -- just reports, so a future negative bin gets noticed
# immediately instead of sitting undiagnosed the way this one did.
# ---------------------------------------------------------------------------

def scan_negative_bins(histograms, include_zx=True):
    print("\n[Negative-bin scan]")
    n_checked, flagged = 0, []
    for name, h in histograms.items():
        if name == "ZX" and not include_zx:
            continue
        n_checked += 1
        neg = np.where(h < 0)[0]
        if len(neg):
            flagged.append((name, [(int(i), float(h[i])) for i in neg], float(h.sum())))
    print(f"  Histograms checked: {n_checked}  |  with negative bins: {len(flagged)}")
    for name, bins, total in flagged:
        print(f"    h_{name}: total={total:.4e}  negative_bins={bins}")
    if not flagged:
        print("  Clean -- no negative bins in any process/systematic template.")
    return flagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build combine histograms and datacards from test-set scored parquets"
    )
    parser.add_argument(
        "--scored-dir",
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored_v4",
        help="Directory with MVA-scored parquets from the ctag2d WORKFLOW VARIANT "
             "(hplusc_mva_4class_ctag2d.yaml) -- NOT the production hplusc_mva_4class "
             "scored dir, which never has a weight_CMS_ctag2d_* column. Does not exist "
             "until that workflow has been run + scored. _v4 (default since 2026-08-29) "
             "is the first rebuild built on a genuinely complete production tree AND the "
             "first to carry the lhescale.py Up/Down-swap fix (see README_HcZZ.md "
             "2026-08-29 dated entry). _v2/_v3 predate that fix and silently reproduce "
             "a stale swapped-label result.",
    )
    parser.add_argument(
        "--sumw-dir",
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d",
        help="Base parquet production directory (ctag2d workflow variant) containing "
             "<era>/<dataset>/sumw/*.json sidecars (true pre-selection sum of genWeight "
             "per chunk)",
    )
    parser.add_argument(
        "--zx-dir",
        default="/eos/user/s/snandaku/Analysis/zx_background_mva_os_90160_featmajor",
        help="Directory with ZX MVA parquets (zx_background_mva_ss or _os). Default "
             "fixed 2026-08-17: the inherited default (from create_datacards.py's own "
             "'_ss_90160' -- same-sign) does NOT match what every documented reference "
             "result in this repo was actually built with (OS/opposite-sign, "
             "'_featmajor'). Confirmed the hard way: with the old SS default, the "
             "otherwise-identical --original preset in ../run_pipeline.sh silently gives "
             "r=364.5, not the documented r=317.0. create_datacards.py itself (the "
             "separate 1D-SF production script, untouched) still carries the stale SS "
             "default -- this divergence is deliberate, ctag2d-pipeline-local only.",
    )
    parser.add_argument(
        "--output",
        default="/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d",
        help="Output directory for ROOT file and datacard (ctag2d variant -- kept "
             "separate from the production combine_run3_90160/ dir). Note: this "
             "deliberately diverges from create_datacards.py's own default "
             "(/eos/user/s/snandaku/Analysis/combine_run3_90160_v5), which still follows "
             "the project's long-standing Analysis/combine_run3_* results-archive "
             "convention documented in README_HcZZ.md -- ctag2d's output was moved under "
             "Higgscharmnew/higgscharm/combine/ instead, by explicit user request 2026-08-16, to keep "
             "it colocated with the combine software build it was produced with.",
    )
    parser.add_argument(
        "--binning",
        choices=["equal", "quantile"],
        default="quantile",
        help="'equal': fixed 20 equal-width [0,1] bins (old default, wastes most "
             "bins on the mva_score_Signal distribution's low-score pile-up). "
             "'quantile': variable-width bins from qqZZ-weighted quantiles "
             "(default) -- see compute_quantile_bins().",
    )
    parser.add_argument(
        "--n-bins", type=int, default=20,
        help="Number of bins (equal-width) or quantile bins requested (quantile "
             "may return fewer after deduping degenerate edges).",
    )
    parser.add_argument(
        "--merge-bin-ranges", default=None,
        help="'Surgical rebin' (2026-08-17): comma-separated inclusive bin-index "
             "ranges (0-based, into the just-computed --n-bins quantile edges) to "
             "merge into single wide bins, e.g. '2-4,7-10'. Fixes the negative-bin "
             "issue found in Signal's low-score-tail templates at zero measured "
             "sensitivity cost (see outputs/combine_run3_90160_ctag2d_negbin_"
             "rebinning/README.md). Empirically tuned for --n-bins 20 on the current "
             "data -- re-diagnose (scripts/01-05 in that README) if the underlying "
             "parquets change enough to shift the quantile edges. Default: no "
             "merging (original behavior, may contain negative bins).",
    )
    parser.add_argument(
        "--skip-zx", action="store_true",
        help="Exclude Z+X (reducible background) entirely: don't read --zx-dir, no "
             "ZX process/row/rateParam in the datacard. Writes datacard_no_zx.txt "
             "instead of datacard_mva_with_zx.txt.",
    )
    parser.add_argument(
        "--automcstats-threshold", type=int, default=None,
        help="autoMCStats effective-event-count cutoff. OFF by default (2026-08-19) "
             "-- confirmed non-convergent at every threshold tested (10/50/100, "
             "11.5h each, zero quantiles produced), see the datacard's own inline "
             "comment above the autoMCStats line and README_HcZZ.md. Pass a value "
             "to opt back in for testing.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("H+c → ZZ → 4l  |  Combine datacard builder")
    print(f"Mass window : {MASS_WINDOW[0]}–{MASS_WINDOW[1]} GeV")
    print(f"Total lumi  : {TOTAL_LUMI_PB:.1f} pb^-1")
    print(f"Binning     : {args.binning} ({args.n_bins} bins requested)")
    print("=" * 70)

    # 1a. True pre-selection sumw per process (for correct xs*lumi normalization)
    print(f"\n[1a] Pre-selection sumw\n    Dir: {args.sumw_dir}")
    sumw_by_proc = load_sumw(args.sumw_dir)

    # 1b. MC: shapes + normalization from all scored parquets (xs × lumi scaled)
    print(f"\n[1b] MC scored parquets\n    Dir: {args.scored_dir}")
    mc_data = load_mc_scored_parquets(args.scored_dir, sumw_by_proc)

    # 1c. Bin edges -- must happen after mc_data is loaded (quantile binning
    # is data-driven) and before any histogram is filled.
    global MVA_BINS
    if args.binning == "quantile":
        MVA_BINS = compute_quantile_bins(mc_data, n_bins=args.n_bins)
    else:
        MVA_BINS = np.linspace(0, 1, args.n_bins + 1)
    print(f"\n[1c] Bin edges ({len(MVA_BINS)-1} bins):\n    {np.round(MVA_BINS, 4)}")

    if args.merge_bin_ranges:
        merge_ranges = parse_merge_ranges(args.merge_bin_ranges)
        MVA_BINS = merge_bin_edges(MVA_BINS, merge_ranges)
        print(f"\n[1c'] Surgical rebin applied ({args.merge_bin_ranges}) -> "
              f"{len(MVA_BINS)-1} bins:\n    {np.round(MVA_BINS, 4)}")

    # 2. Z+X
    if args.skip_zx:
        print("\n[2] Z+X background: SKIPPED (--skip-zx)")
        zx_scores, zx_weights = np.array([]), np.array([])
    else:
        print(f"\n[2] Z+X background\n    Dir: {args.zx_dir}")
        zx_scores, zx_weights = load_zx_parquets(args.zx_dir)

    # 3. Build histograms
    print("\n[3] Building histograms")
    histograms, histograms_var = build_histograms(mc_data, zx_scores, zx_weights)
    if args.skip_zx:
        histograms.pop("ZX", None)
        histograms_var.pop("ZX", None)

    # 4. Write output
    print("\n[4] Writing output")
    root_filename = "histograms_mva_with_zx.root" if not args.skip_zx else "histograms_no_zx.root"
    root_file = write_root_file(histograms, histograms_var, args.output, filename=root_filename, include_zx=not args.skip_zx)
    dc_path = write_datacard(histograms, os.path.basename(root_file), args.output, include_zx=not args.skip_zx, automcstats_threshold=args.automcstats_threshold)

    print_summary(histograms, include_zx=not args.skip_zx)
    scan_negative_bins(histograms, include_zx=not args.skip_zx)

    print(f"\n[Done]  All files in: {args.output}")
    print(f"\nTo run combine:")
    print(f"  cd {args.output}")
    print(f"  combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001 --rRelAcc 0.00001 \\")
    print(f"    {os.path.basename(dc_path)} -n HcZZ_ctag2d{'_no_ZX' if args.skip_zx else '_with_ZX'}")


if __name__ == "__main__":
    main()
