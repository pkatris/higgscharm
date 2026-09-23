#!/usr/bin/env python3
"""
qqZZ CR (m4l > 180 GeV) / On-Z CR (|m4l - 91.1876| < 7 GeV) ctag2d 2D-SF
validation check -- the full-SR-scope counterpart to
combine/scripts/create_cr_zplusc_ctag2d_check.py (which was limited to the
`cjets`-only subset stored in the Z+c CR parquets, see second-brain memory
hczz_zplusc_cr_ctag2d_check).

Purpose: apply the official PNet 2D ctag SF (CTag2DCorrector,
analysis/corrections/ctag2d.py) to the qqZZ/On-Z CR and compare Data vs MC
central values in each m4l bin, with and without the SF, to see whether the
SF measurably improves or degrades Data/MC agreement. Two SEPARATE
histograms per region (no-SF, with-SF), same binning, per user direction
(same pattern as the Z+c CR check).

Built from the EXISTING `hplusc_mva_4class_ctag2d_scored` production tree's
`base` category (m4l > 70 GeV only, no upper cut at processing time) -- same
tree used by ../../../../../b-hive_ttcc/combine/create_cr_qqzz_onz_yields.py
for the plain (no-SF-check) yields, and the SAME production
`create_datacards_ctag2d_100150_rawsf.py` uses for the real analysis. No new
condor production needed. A CR must be built from an m4l selection only,
orthogonal to the MVA-score fit discriminant (second-brain memory
hczz_cr_sideband_strategy).

KEY WRINKLE (opposite of the Z+c CR check): this production's
`weight_nominal` ALREADY has the central ctag2d SF baked in (confirmed via
`create_datacards_ctag2d_100150_rawsf.py`'s own docstring: "weight_nominal
already has the central ctag2d SF folded in"). So:
  - "with SF"    = weight_nominal, AS-IS.
  - "without SF" = weight_nominal / sf_c, where sf_c is a freshly
    recomputed per-event product SF over the up-to-3-leading jets of ANY
    flavor (`jet_btagPNetCvL/CvB/hadronFlavour/pt`), reusing
    `create_datacards_ctag2d_100150_rawsf.py`'s exact `_raw_event_sf`
    per-jet recipe + product-over-jets convention -- the full SR scope
    (`events.selected_jets`), NOT the cjets-only subset the Z+c CR check
    used.

MC normalization: xs * lumi * (raw_weight_in_bin / sumw_generated), same
convention as create_cr_qqzz_onz_yields.py / create_datacards_ctag2d.py.
Data: raw unweighted counts.

Usage:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 create_cr_qqzz_onz_ctag2d_check.py
"""
import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict
import pyarrow.parquet as pq

HIGGSCHARM_DIR = "/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm"
sys.path.insert(0, HIGGSCHARM_DIR)
from analysis.corrections.ctag2d import _category_np, CAT_TO_WP, CORRECTION_NAME, PT_BINS
from analysis.corrections.utils import correction_files
import correctionlib

sys.path.insert(0, "/eos/home-s/snandaku/b-hive_ttcc/combine")
from create_datacards_ctag2d import (
    ERAS, TOTAL_LUMI_PB, PROCESS_XS_PB, EXPECTED_YIELDS,
    PROCESS_MAPPING, get_process_from_path, load_sumw,
)

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
SUMW_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d"

BKG_PROCS = ["qqZZ", "ggZZ", "Signal", "Other_Higgs"]

ONZ_LO, ONZ_HI = 91.1876 - 7, 91.1876 + 7
QQZZ_CR_LO = 180.0

DATA_PREFIXES = ("EGamma", "Muon")

ONZ_HIST_EDGES = np.linspace(76, 106, 16)      # 2 GeV bins, [76,106)
QQZZ_HIST_EDGES = np.linspace(160, 400, 25)    # 10 GeV bins, [160,400)

NJ_CAP = 3  # matches CTag2DCorrector / SR scope: up to 3 leading jets, any flavor

OUT_JSON = "/tmp/snandaku/claude-178308/-eos-home-s-snandaku-second-brain/394d3520-207d-4ee7-803d-8c8427b68238/scratchpad/cr_qqzz_onz_ctag2d_check.json"

_corr_cache = {}


def get_corr(era):
    if era not in _corr_cache:
        _corr_cache[era] = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][era]
        )[CORRECTION_NAME]
    return _corr_cache[era]


def _raw_event_sf_flat(cvl, cvb, flav, pt, era, syst="central"):
    """Per-jet SF, identical recipe to create_datacards_ctag2d_100150_rawsf.py's
    _raw_event_sf (official, unswapped SFs, current corrector state)."""
    corr = get_corr(era)
    cat = _category_np(cvl, cvb, era)
    wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
    ok = (wp >= 0) & np.isfinite(pt)
    sf = np.ones(len(flav), dtype=np.float64)
    if ok.any():
        sf[ok] = corr.evaluate(
            syst, flav[ok].astype(np.int64), wp[ok],
            np.zeros(ok.sum(), dtype=np.float64),
            np.clip(pt[ok], *PT_BINS),
        )
    return sf


def _event_central_sf(df, era):
    """Per-event product of the central per-jet SF over up to NJ_CAP leading
    jets of any flavor -- the same scope/convention as CTag2DCorrector."""
    n = len(df)
    cvl_list = df["jet_btagPNetCvL"].values
    cvb_list = df["jet_btagPNetCvB"].values
    flav_list = df["jet_hadronFlavour"].values
    pt_list = df["jet_pt"].values

    nj = np.array([len(x[:NJ_CAP]) for x in cvl_list])
    if n == 0 or nj.sum() == 0:
        return np.ones(n, dtype=np.float64)

    cvl = np.concatenate([np.asarray(x[:NJ_CAP], dtype=np.float64) for x in cvl_list])
    cvb = np.concatenate([np.asarray(x[:NJ_CAP], dtype=np.float64) for x in cvb_list])
    flav = np.concatenate([np.asarray(x[:NJ_CAP], dtype=np.int64) for x in flav_list])
    pt = np.concatenate([np.asarray(x[:NJ_CAP], dtype=np.float64) for x in pt_list])

    sf_flat = _raw_event_sf_flat(cvl, cvb, flav, pt, era, "central")

    out = np.ones(n, dtype=np.float64)
    idx = 0
    for i, k in enumerate(nj):
        if k > 0:
            out[i] = np.prod(sf_flat[idx:idx + k])
        idx += k
    return out


def _unwrap_m4l(series):
    return np.array([x[0] if len(x) else np.nan for x in series.values])


def is_data_sample(sample_name):
    return get_process_from_path(sample_name) is None and sample_name.startswith(DATA_PREFIXES)


def load_region_yields(sumw_by_proc):
    # keyed by proc -> {"onz":..., "qqzzcr":...} raw (unscaled) weight sums
    mc_raw_nosf = defaultdict(lambda: {"onz": 0.0, "qqzzcr": 0.0})
    mc_raw_withsf = defaultdict(lambda: {"onz": 0.0, "qqzzcr": 0.0})
    mc_n = defaultdict(lambda: {"onz": 0, "qqzzcr": 0})
    data_n = {"onz": 0, "qqzzcr": 0}

    mc_hist_onz_nosf = defaultdict(lambda: np.zeros(len(ONZ_HIST_EDGES) - 1))
    mc_hist_onz_withsf = defaultdict(lambda: np.zeros(len(ONZ_HIST_EDGES) - 1))
    mc_hist_qqzzcr_nosf = defaultdict(lambda: np.zeros(len(QQZZ_HIST_EDGES) - 1))
    mc_hist_qqzzcr_withsf = defaultdict(lambda: np.zeros(len(QQZZ_HIST_EDGES) - 1))
    data_hist_onz = np.zeros(len(ONZ_HIST_EDGES) - 1)
    data_hist_qqzzcr = np.zeros(len(QQZZ_HIST_EDGES) - 1)

    n_mc_files = n_data_files = n_read_errors = 0

    for era in ERAS:
        era_dir = os.path.join(SCORED_DIR, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue
        for sample_name in sorted(os.listdir(era_dir)):
            sample_dir = os.path.join(era_dir, sample_name, "base")
            if not os.path.isdir(sample_dir):
                continue

            proc = get_process_from_path(sample_name)
            data = is_data_sample(sample_name)
            if proc is None and not data:
                continue

            for pq_file in glob.glob(os.path.join(sample_dir, "*.parquet")):
                cols = ["m4l"] if data else [
                    "m4l", "weight_nominal",
                    "jet_btagPNetCvL", "jet_btagPNetCvB", "jet_hadronFlavour", "jet_pt",
                ]
                try:
                    df = pq.read_table(pq_file, columns=cols).to_pandas()
                except Exception:
                    n_read_errors += 1
                    continue

                mass = _unwrap_m4l(df["m4l"])
                valid = ~np.isnan(mass)
                mass = mass[valid]
                if len(mass) == 0:
                    continue

                onz_mask = (mass > ONZ_LO) & (mass < ONZ_HI)
                qqzzcr_mask = mass > QQZZ_CR_LO

                if data:
                    n_data_files += 1
                    data_n["onz"] += int(onz_mask.sum())
                    data_n["qqzzcr"] += int(qqzzcr_mask.sum())
                    h1, _ = np.histogram(mass[(mass > 76) & (mass < 106)], bins=ONZ_HIST_EDGES)
                    h2, _ = np.histogram(mass[(mass >= 160) & (mass < 400)], bins=QQZZ_HIST_EDGES)
                    data_hist_onz += h1
                    data_hist_qqzzcr += h2
                else:
                    n_mc_files += 1
                    df_valid = df[valid].reset_index(drop=True)
                    wgt_withsf = df_valid["weight_nominal"].fillna(0).to_numpy()
                    sf_c = _event_central_sf(df_valid, era)
                    wgt_nosf = np.where(sf_c != 0, wgt_withsf / sf_c, wgt_withsf)

                    mc_raw_withsf[proc]["onz"] += wgt_withsf[onz_mask].sum()
                    mc_raw_withsf[proc]["qqzzcr"] += wgt_withsf[qqzzcr_mask].sum()
                    mc_raw_nosf[proc]["onz"] += wgt_nosf[onz_mask].sum()
                    mc_raw_nosf[proc]["qqzzcr"] += wgt_nosf[qqzzcr_mask].sum()
                    mc_n[proc]["onz"] += int(onz_mask.sum())
                    mc_n[proc]["qqzzcr"] += int(qqzzcr_mask.sum())

                    m_onz_range = (mass > 76) & (mass < 106)
                    m_qqzzcr_range = (mass >= 160) & (mass < 400)
                    h1w, _ = np.histogram(mass[m_onz_range], bins=ONZ_HIST_EDGES, weights=wgt_withsf[m_onz_range])
                    h1n, _ = np.histogram(mass[m_onz_range], bins=ONZ_HIST_EDGES, weights=wgt_nosf[m_onz_range])
                    h2w, _ = np.histogram(mass[m_qqzzcr_range], bins=QQZZ_HIST_EDGES, weights=wgt_withsf[m_qqzzcr_range])
                    h2n, _ = np.histogram(mass[m_qqzzcr_range], bins=QQZZ_HIST_EDGES, weights=wgt_nosf[m_qqzzcr_range])
                    mc_hist_onz_withsf[proc] += h1w
                    mc_hist_onz_nosf[proc] += h1n
                    mc_hist_qqzzcr_withsf[proc] += h2w
                    mc_hist_qqzzcr_nosf[proc] += h2n

    print(f"\n  Files read: {n_mc_files} MC, {n_data_files} data, {n_read_errors} read errors")
    return (mc_raw_nosf, mc_raw_withsf, mc_n, data_n,
             mc_hist_onz_nosf, mc_hist_onz_withsf,
             mc_hist_qqzzcr_nosf, mc_hist_qqzzcr_withsf,
             data_hist_onz, data_hist_qqzzcr)


def main():
    print("Loading sumw sidecars (ctag2d workflow)...")
    sumw_by_proc = load_sumw(SUMW_DIR)
    for p in BKG_PROCS:
        print(f"  sumw[{p}] = {sumw_by_proc.get(p, 0.0):.4g}")

    print(f"\nOn-Z CR window: [{ONZ_LO:.2f}, {ONZ_HI:.2f}] GeV")
    print(f"qqZZ CR window: m4l > {QQZZ_CR_LO:.1f} GeV")
    print("\nScanning production tree (this reads every base-category parquet, all 4 eras)...")

    (mc_raw_nosf, mc_raw_withsf, mc_n, data_n,
     mc_hist_onz_nosf, mc_hist_onz_withsf,
     mc_hist_qqzzcr_nosf, mc_hist_qqzzcr_withsf,
     data_hist_onz, data_hist_qqzzcr) = load_region_yields(sumw_by_proc)

    scales = {}
    for proc in BKG_PROCS:
        sumw_gen = sumw_by_proc.get(proc, 0.0)
        exp = EXPECTED_YIELDS.get(proc, 0.0)
        scales[proc] = exp / sumw_gen if sumw_gen > 0 else float("nan")

    def total(mc_raw, region):
        return sum(mc_raw[p][region] * scales[p] for p in BKG_PROCS)

    def ratio(d, m):
        return d / m if m > 0 else float("nan")

    print("\n" + "=" * 90)
    print(f"{'Region':<14}{'Data':>10}{'MC noSF':>14}{'D/MC noSF':>12}{'MC withSF':>14}{'D/MC withSF':>14}")
    print("-" * 90)
    for region, label in [("onz", "On-Z CR"), ("qqzzcr", "qqZZ CR (>180)")]:
        d = data_n[region]
        m0 = total(mc_raw_nosf, region)
        m1 = total(mc_raw_withsf, region)
        print(f"{label:<14}{d:>10d}{m0:>14.3f}{ratio(d, m0):>12.3f}{m1:>14.3f}{ratio(d, m1):>14.3f}")

    print("\nPer-process breakdown:")
    for region, label in [("onz", "On-Z CR"), ("qqzzcr", "qqZZ CR (>180)")]:
        print(f"\n  --- {label} ---")
        for proc in BKG_PROCS:
            y0 = mc_raw_nosf[proc][region] * scales[proc]
            y1 = mc_raw_withsf[proc][region] * scales[proc]
            delta = (y1 - y0) / y0 * 100 if y0 else float("nan")
            print(f"    {proc:<14} noSF={y0:>10.3f}  withSF={y1:>10.3f}  delta={delta:>+7.2f}%  (n={mc_n[proc][region]})")

    print("\nPer-bin comparison (m4l bin -> data, MC noSF, MC withSF):")
    for region, edges, dhist, h_nosf, h_withsf in [
        ("On-Z CR", ONZ_HIST_EDGES, data_hist_onz, mc_hist_onz_nosf, mc_hist_onz_withsf),
        ("qqZZ CR", QQZZ_HIST_EDGES, data_hist_qqzzcr, mc_hist_qqzzcr_nosf, mc_hist_qqzzcr_withsf),
    ]:
        print(f"\n  --- {region} ---")
        mc_tot_nosf = np.zeros(len(edges) - 1)
        mc_tot_withsf = np.zeros(len(edges) - 1)
        for p in BKG_PROCS:
            mc_tot_nosf += h_nosf[p] * scales[p]
            mc_tot_withsf += h_withsf[p] * scales[p]
        header = f"{'bin_lo':>8}{'bin_hi':>8}{'data':>8}{'MC_noSF':>12}{'D/MC_noSF':>12}{'MC_withSF':>12}{'D/MC_withSF':>14}"
        print(f"    {header}")
        for i in range(len(edges) - 1):
            d = dhist[i]
            m0 = mc_tot_nosf[i]
            m1 = mc_tot_withsf[i]
            print(f"    {edges[i]:>8.1f}{edges[i+1]:>8.1f}{d:>8.0f}{m0:>12.3f}{ratio(d,m0):>12.3f}{m1:>12.3f}{ratio(d,m1):>14.3f}")

    out = {
        "onz_window": [ONZ_LO, ONZ_HI],
        "qqzzcr_lo": QQZZ_CR_LO,
        "onz_edges": ONZ_HIST_EDGES.tolist(),
        "qqzzcr_edges": QQZZ_HIST_EDGES.tolist(),
        "data_hist_onz": data_hist_onz.tolist(),
        "data_hist_qqzzcr": data_hist_qqzzcr.tolist(),
        "mc_hist_onz_nosf": {p: (mc_hist_onz_nosf[p] * scales[p]).tolist() for p in BKG_PROCS},
        "mc_hist_onz_withsf": {p: (mc_hist_onz_withsf[p] * scales[p]).tolist() for p in BKG_PROCS},
        "mc_hist_qqzzcr_nosf": {p: (mc_hist_qqzzcr_nosf[p] * scales[p]).tolist() for p in BKG_PROCS},
        "mc_hist_qqzzcr_withsf": {p: (mc_hist_qqzzcr_withsf[p] * scales[p]).tolist() for p in BKG_PROCS},
        "yields": {
            "onz_nosf": {p: mc_raw_nosf[p]["onz"] * scales[p] for p in BKG_PROCS},
            "onz_withsf": {p: mc_raw_withsf[p]["onz"] * scales[p] for p in BKG_PROCS},
            "qqzzcr_nosf": {p: mc_raw_nosf[p]["qqzzcr"] * scales[p] for p in BKG_PROCS},
            "qqzzcr_withsf": {p: mc_raw_withsf[p]["qqzzcr"] * scales[p] for p in BKG_PROCS},
            "data_onz": data_n["onz"],
            "data_qqzzcr": data_n["qqzzcr"],
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote JSON sidecar: {OUT_JSON}")


if __name__ == "__main__":
    main()
