"""PNet / UParT 2D pseudo-continuous HF-tagging scale factor.

Preliminary "2D_HF_Tagging" SF (correction_files["ctagging_2d"]), distinct from the 1D
fixed-WP CTagCorrector (BTV particleNet_wc/tnp/light, ctag.py) this is meant to replace.
Tagger per era: ParticleNetAK4_pseudocontinuous (btagPNetCvL/CvB) for 2022/2023,
UParTAK4_pseudocontinuous (btagUParTAK4CvL/CvB) for 2024 -- see TAGGER_BY_ERA.

**2024 SUPPORT ADDED 2026-10-21**: 2024 UParTAK4 category boundaries added to
BOUNDARIES_BY_ERA from the 2D HF-tagging WP-definition table (score(HF vs. LF) /
score(B vs. C) columns: L0 <=0.250; C0 (0.250,0.452]; C1 (0.452,0.808]; C2-C4 and B0-B4
at HFvLF>0.808 split in BvC at 0.006/0.017/0.055/0.761/0.944/0.985/0.995). Same
(x, y) transform as 2022/2023 (x=HFvLF from CvL/CvB, y=BvC=1-CvB), but fed with the
UParT discriminants since the 2024 SF file is calibrated for UParTAK4.
Also fixed: jets with tagger default scores (e.g. -1, so y=1-CvB=2 lands outside [0,1])
were caught by the closed-corner B4 fallback and given B4 SFs; inputs are now required to
lie in [0,1]x[0,1], and out-of-range jets get SF=1 as originally intended.

WP-code mapping (0=L0, 40-44=C0-C4, 50-54=B0-B4) and the (syst, flavor, wp, abseta, pt)
evaluate order were ported from `Chirayu18/higgscharm` (`hww-analysis` branch,
analysis/corrections/ctag2d.py) -- that repo runs the identical SF files
(flavTaggingSF_<campaign>.json.gz, correction name ParticleNetAK4_pseudocontinuous) in
production. This resolved the WP-code documentation blocker recorded in second-brain memory
hczz_pnet_ctag_sf_migration (the etsai.web.cern.ch reference doc is CERN-SSO-gated).

**Category boundaries, CORRECTED 2026-08-14** (originally shipped 2026-08-14 with a bug,
fixed same day): the FIRST version of this file used the hww-analysis reference's edges
verbatim -- confirmed after the fact (user pasted the real etsai.web.cern.ch page content) to
be the 2024 UParTAK4 boundaries, NOT the 2022/2023 PNet boundaries this file's SF_FILES
actually need. Real per-era PNet boundaries (source: HiggsDNA metaconditions Era2022_v1.json
"HPC_ctag_WPs" block, independently confirmed 2026-08-14 by the user pasting that exact JSON)
were already correctly implemented in this repo at
`b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::JET_TAG_BOUNDARIES_BY_ERA` --
copied from there below rather than re-derived.

The (x, y) coordinate transform itself was NOT the bug and needed no fix: `lz4_hczz_processor.
py::compute_jet_categories` derives x=pBplusC=B+(1-B)*CvL, y=pBvsC=1-CvB from the B/CvB/CvL
softmax outputs; this file computes x=CvL/(CvL+CvB*(1-CvL)), y=1-CvB from CvL/CvB alone --
algebraically identical (substitute CvB=C/(B+C), CvL=C/(C+L) into either and both reduce to
(B+C)/(B+C+L) for x), confirmed against the actual pasted etsai.web.cern.ch UParT-doc HFvLF
formula too. Only the per-era EDGE VALUES were wrong, not the transform.

SCOPE DECISION (differs from hww-analysis): that repo applies this to a single "event
candidate c-jet" (their MVA is keyed on one leading-CvsL jet). This analysis's existing 1D
CTagCorrector (ctag.py) instead multiplies a per-jet SF across every jet in
`events.selected_jets` (all flavors, tagged + untagged formula). To keep scope identical to
the corrector being replaced -- so yields are comparable before/after and c-tagging remains
tied to the full selected-jet collection, not one jet -- this version evaluates a 2D category
per jet and takes ak.prod across ALL selected jets in the event. A pseudo-continuous SF does
not need ctag.py's separate tagged/untagged eff-based formula: every jet (including L0,
"fails the tag") gets its own category's data/MC correction factor.

CONFIRMED 2026-08-18: this all-jets scope is the source of the CMS_ctag2d Down-variation
asymmetry (placeholder central=1.000/up_Total=3.000/down_Total=0.300 calibration values in
statistically-empty (flavor, WP) corners of the official JSON, amplified by ak.prod across
many jets). User sign-off to keep this scope regardless -- yield-comparability with the
corrector being replaced outweighs the smoother-but-incomparable one-jet alternative. Not a
bug; documented caveat only (see README_HcZZ.md "Known traps").

**SCOPE REVISITED 2026-08-31**: the 2026-08-18 sign-off was made against a coarse
freeze-and-compare impact of 71.2 (rank 1). The real `combineTool.py -M Impacts`
profile-likelihood fit on the deduplicated `_v5` card (2026-08-30) found CMS_ctag2d's
real impact is 143.6 -- ~3.2x the #2 nuisance -- roughly double what the original
sign-off was evaluated against. Root-cause quantified directly: the official
`flavTaggingSF_<year>.json.gz` returns the exact placeholder triple
(central=1.000, up_Total=3.000, down_Total=0.300) for 41% of the (flavor, WP, pt)
grid queried (2022postEE), not a rare corner. Because `ak.prod` ran over EVERY
`selected_jets` entry (uncapped -- pt>20/|eta|<2.5/jetID/dR-lepton only), an event
with 2 placeholder-corner jets got a per-event Up/nominal ratio up to 9.0 (3.0x3.0)
and Down as low as 0.09 (0.3x0.3) -- verified directly on real qqZZ scored parquets
(692 events, 2022postEE, 1.5% of events with Up/nom>5, up to 9.0x max).

Checked whether HWW's single-c-jet scope (`Chirayu18/higgscharm`, `hww-analysis`
branch) is the right model to copy: NO -- their MVA is keyed on exactly 1 jet, ours
(`models/config_4class.yml`, `n_cpf_candidates: 3`) deliberately looks at up to 3
jets of ANY flavor category (b-jets for H+b, light-jets for VBF/qqZZ, not just
c-jets), so a single-c-jet scope would silently stop correcting jets 2-3's real
tagging-SF mismodeling -- HWW's scope isn't a fix, it's just a narrower model that
was never exposed to this failure mode. The physically-matched fix is a 3-jet cap
(below), not a 1-jet cap: `events.selected_jets` is uncapped multiplicity (jet
multiplicity check, `_scored_v5`, all 4 eras: >3 jets is 0.5% Signal / 1.4% qqZZ /
1.1% ggZZ / 7.6% Other_Higgs of events) -- bounding to the same 3 jets the MVA
actually consumes removes the unbounded tail from 4+-jet events (mostly
Other_Higgs, via ttH/VBF) but does NOT eliminate the core pathology, since most
events already have <=3 jets and can still have 2 of those 3 land in placeholder
corners (the 9.0x qqZZ example above is a <=3-jet process). This is a real,
partial mitigation for physics-consistency reasons, not a full fix for the
placeholder-value compounding itself -- that would need excluding placeholder
cells from the uncertainty specifically, deferred (not requested this session).
Full writeup: second-brain `Tasks/HcZZ-fake-rate.md` "Update 2026-08-31", memory
`hczz_ctag2d_impact_investigation`.
"""
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections.utils import correction_files

CATS = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
CID = {n: i for i, n in enumerate(CATS)}
WP_ID = {"L0": 0, "C0": 40, "C1": 41, "C2": 42, "C3": 43, "C4": 44,
         "B0": 50, "B1": 51, "B2": 52, "B3": 53, "B4": 54}
CAT_TO_WP = np.array([WP_ID[c] for c in CATS], dtype=np.int64)

# Per-era 2D-category boundaries in (pBplusC=HFvLF, pBvsC=BvC=1-CvB) space, i.e.
# {"x": (lo, hi), "y": (lo, hi)}. 2022/2023 (PNet) source: HiggsDNA metaconditions
# Era2022_v1.json "HPC_ctag_WPs" block, cross-confirmed 2026-08-14 against
# b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::JET_TAG_BOUNDARIES_BY_ERA
# (byte-for-byte match). 2024 (UParT) source: 2D HF-tagging WP-definition table
# (added 2026-10-21). wp_id -> category: 0=L0, 40-44=C0-C4, 50-54=B0-B4.
BOUNDARIES_BY_ERA = {
    "2022preEE": {
        "L0": {"x": (0.0,   0.160), "y": (0.0,   1.0)},
        "C0": {"x": (0.160, 0.332), "y": (0.000, 1.000)},
        "C1": {"x": (0.332, 0.706), "y": (0.000, 1.000)},
        "C2": {"x": (0.706, 1.000), "y": (0.090, 0.261)},
        "C3": {"x": (0.706, 1.000), "y": (0.036, 0.090)},
        "C4": {"x": (0.706, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.706, 1.000), "y": (0.261, 0.799)},
        "B1": {"x": (0.706, 1.000), "y": (0.799, 0.933)},
        "B2": {"x": (0.706, 1.000), "y": (0.933, 0.978)},
        "B3": {"x": (0.706, 1.000), "y": (0.978, 0.993)},
        "B4": {"x": (0.706, 1.000), "y": (0.993, 1.000)},
    },
    "2022postEE": {
        "L0": {"x": (0.0,   0.160), "y": (0.0,   1.0)},
        "C0": {"x": (0.160, 0.332), "y": (0.000, 1.000)},
        "C1": {"x": (0.332, 0.706), "y": (0.000, 1.000)},
        "C2": {"x": (0.706, 1.000), "y": (0.090, 0.261)},
        "C3": {"x": (0.706, 1.000), "y": (0.036, 0.090)},
        "C4": {"x": (0.706, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.706, 1.000), "y": (0.261, 0.799)},
        "B1": {"x": (0.706, 1.000), "y": (0.799, 0.932)},
        "B2": {"x": (0.706, 1.000), "y": (0.932, 0.978)},
        "B3": {"x": (0.706, 1.000), "y": (0.978, 0.993)},
        "B4": {"x": (0.706, 1.000), "y": (0.993, 1.000)},
    },
    "2023preBPix": {
        "L0": {"x": (0.0,   0.144), "y": (0.0,   1.0)},
        "C0": {"x": (0.144, 0.292), "y": (0.000, 1.000)},
        "C1": {"x": (0.292, 0.646), "y": (0.000, 1.000)},
        "C2": {"x": (0.646, 1.000), "y": (0.078, 0.240)},
        "C3": {"x": (0.646, 1.000), "y": (0.036, 0.078)},
        "C4": {"x": (0.646, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.646, 1.000), "y": (0.240, 0.752)},
        "B1": {"x": (0.646, 1.000), "y": (0.752, 0.915)},
        "B2": {"x": (0.646, 1.000), "y": (0.915, 0.972)},
        "B3": {"x": (0.646, 1.000), "y": (0.972, 0.992)},
        "B4": {"x": (0.646, 1.000), "y": (0.992, 1.000)},
    },
    "2023postBPix": {
        "L0": {"x": (0.0,   0.144), "y": (0.0,   1.0)},
        "C0": {"x": (0.144, 0.288), "y": (0.000, 1.000)},
        "C1": {"x": (0.288, 0.640), "y": (0.000, 1.000)},
        "C2": {"x": (0.640, 1.000), "y": (0.082, 0.243)},
        "C3": {"x": (0.640, 1.000), "y": (0.036, 0.082)},
        "C4": {"x": (0.640, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.640, 1.000), "y": (0.243, 0.742)},
        "B1": {"x": (0.640, 1.000), "y": (0.742, 0.909)},
        "B2": {"x": (0.640, 1.000), "y": (0.909, 0.969)},
        "B3": {"x": (0.640, 1.000), "y": (0.969, 0.992)},
        "B4": {"x": (0.640, 1.000), "y": (0.992, 1.000)},
    },
    "2024": {
        "L0": {"x": (0.0,   0.250), "y": (0.0,   1.0)},
        "C0": {"x": (0.250, 0.452), "y": (0.000, 1.000)},
        "C1": {"x": (0.452, 0.808), "y": (0.000, 1.000)},
        "C2": {"x": (0.808, 1.000), "y": (0.017, 0.055)},
        "C3": {"x": (0.808, 1.000), "y": (0.006, 0.017)},
        "C4": {"x": (0.808, 1.000), "y": (0.000, 0.006)},
        "B0": {"x": (0.808, 1.000), "y": (0.055, 0.761)},
        "B1": {"x": (0.808, 1.000), "y": (0.761, 0.944)},
        "B2": {"x": (0.808, 1.000), "y": (0.944, 0.985)},
        "B3": {"x": (0.808, 1.000), "y": (0.985, 0.995)},
        "B4": {"x": (0.808, 1.000), "y": (0.995, 1.000)},
    },
}

# (correction name in the SF file, CvL branch, CvB branch) per era
TAGGER_BY_ERA = {
    "2022preEE": ("ParticleNetAK4_pseudocontinuous", "btagPNetCvL", "btagPNetCvB"),
    "2022postEE": ("ParticleNetAK4_pseudocontinuous", "btagPNetCvL", "btagPNetCvB"),
    "2023preBPix": ("ParticleNetAK4_pseudocontinuous", "btagPNetCvL", "btagPNetCvB"),
    "2023postBPix": ("ParticleNetAK4_pseudocontinuous", "btagPNetCvL", "btagPNetCvB"),
    "2024": ("UParTAK4_pseudocontinuous", "btagUParTAK4CvL", "btagUParTAK4CvB"),
}
NUIS_YEAR = {"2022preEE": "2022", "2022postEE": "2022",
             "2023preBPix": "2023", "2023postBPix": "2023",
             "2024": "2024"}
# single-nuisance source keys, matching the hww-analysis reference (decision there: 2026-07-23,
# Total; per-source decorrelation deferred to a whole-card decorrelation pass -- same deferral
# applies here, not yet revisited).
SYST_UP = "up_Total"
SYST_DN = "down_Total"
PT_BINS = (20.0001, 9999.0)


def _category_np(cvsl, cvsb, era):
    """Map (CvL, CvB) per jet to one of the 11 WP-code categories, using ERA-SPECIFIC
    boundaries (they differ meaningfully between eras -- see BOUNDARIES_BY_ERA).
    -1 = invalid tagger input (NaN or outside [0,1], e.g. default -1 scores) -> SF=1."""
    den = cvsl + cvsb * (1.0 - cvsl)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(den != 0, cvsl / den, np.nan)
    y = 1.0 - cvsb
    good = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    )
    cat = np.full(x.shape, -1, dtype=np.int64)
    boundaries = BOUNDARIES_BY_ERA[era]
    for name, bounds in boundaries.items():
        x_lo, x_hi = bounds["x"]
        y_lo, y_hi = bounds["y"]
        m = good & (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
        cat[m] = CID[name]
    # closed upper edges (x=1 and/or y=1) fall just outside every "< hi" box -- assign them
    # to the category owning that edge. y=1 at high x -> B4; x=1 at any y -> the high-x box
    # containing y (handled by re-running the high-x boxes with x<=1 inclusive).
    unassigned = good & (cat < 0)
    if unassigned.any():
        for name, bounds in boundaries.items():
            x_lo, x_hi = bounds["x"]
            y_lo, y_hi = bounds["y"]
            m = (
                unassigned & (x >= x_lo) & (x <= x_hi)
                & (y >= y_lo) & ((y < y_hi) | ((y_hi == 1.0) & (y <= y_hi)))
            )
            cat[m & (cat < 0)] = CID[name]
    return cat


class CTag2DCorrector:
    """
    2D pseudo-continuous c-tag corrector, applied to the (up to 3) leading jets in
    `events.selected_jets` (product across the event).

    Parameters:
    -----------
        events:
            Events collection (must carry `selected_jets` with the era's CvL/CvB branches
            -- see TAGGER_BY_ERA -- and hadronFlavour)
        weights:
            Weights container from coffea.analysis_tools
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023postBPix, 2024}
        variation:
            if 'nominal' (default), add nominal + up/down variations; else nominal only
            (matches ctag.py's convention in this repo -- NOT hww-analysis's `shift is None`
            polarity, which is inverted)
    """

    def __init__(self, events, weights: Type[Weights], year: str, variation: str = "nominal") -> None:
        if year not in BOUNDARIES_BY_ERA or year not in TAGGER_BY_ERA:
            raise ValueError(
                f"CTag2DCorrector has no category boundaries/tagger config for year '{year}' "
                f"(only {list(BOUNDARIES_BY_ERA)} are wired)."
            )
        self._year = year
        self._weights = weights
        self._variation = variation
        corr_name, cvl_branch, cvb_branch = TAGGER_BY_ERA[year]
        cset = correctionlib.CorrectionSet.from_file(correction_files["ctagging_2d"][year])
        if corr_name not in cset:
            raise KeyError(
                f"'{corr_name}' not found in {correction_files['ctagging_2d'][year]} "
                f"(available: {list(cset.keys())})"
            )
        self._corr = cset[corr_name]
        # evaluate() below assumes inputs (systematic, flavor, wp, abseta, pt)
        input_names = [i.name for i in self._corr.inputs]
        if len(input_names) != 5:
            raise ValueError(
                f"{corr_name} ({year}) has inputs {input_names}; expected 5 "
                f"(systematic, flavor, wp, abseta, pt)"
            )
        self._nuis = f"CMS_ctag2d_{NUIS_YEAR[year]}"

        # CAPPED 2026-08-31: at most the 3 leading (pT-descending, post-selection --
        # NanoAOD's native Jet order, preserved through the pt/eta/jetID/dR cuts in
        # the workflow yaml) jets, matching `n_cpf_candidates: 3` in
        # models/config_4class.yml -- the MVA itself only ever looks at 3 jets'
        # tagging categories (any flavor, not just c-jets), so those are the only
        # jets whose tagging SF mismodeling can actually propagate into the
        # analysis. `events.selected_jets` is NOT capped (pt>20/|eta|<2.5/jetID/
        # dR-lepton only, unbounded multiplicity) -- see the "3-jet cap" note above
        # for why the previous all-jets ak.prod scope was wrong, not just imprecise.
        jets = events.selected_jets[:, :3]
        self._nj = ak.num(jets)
        j = ak.flatten(jets)
        self._cvsl = ak.to_numpy(ak.fill_none(j[cvl_branch], np.nan)).astype(np.float64)
        self._cvsb = ak.to_numpy(ak.fill_none(j[cvb_branch], np.nan)).astype(np.float64)
        self._pt = ak.to_numpy(ak.fill_none(j.pt, np.nan)).astype(np.float64)
        flav = ak.fill_none(getattr(j, "hadronFlavour", ak.zeros_like(j.pt)), 0)
        self._flav = ak.to_numpy(flav).astype(np.int64)

    def _eval_flat(self, syst):
        n = len(self._flav)
        sf = np.ones(n, dtype=np.float64)
        cat = _category_np(self._cvsl, self._cvsb, self._year)
        wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
        ok = (wp >= 0) & np.isfinite(self._pt)
        if ok.any():
            sf[ok] = self._corr.evaluate(
                syst,
                self._flav[ok],
                wp[ok],
                np.zeros(ok.sum(), dtype=np.float64),  # abseta -- inclusive, per reference impl
                np.clip(self._pt[ok], *PT_BINS),
            )
        return sf

    def _eval_event(self, syst):
        """Per-jet SF -> per-event product (ak.prod over jets, matching ctag.py's
        tagged_sf/untagged_sf multiplicative combination)."""
        flat = self._eval_flat(syst)
        return ak.to_numpy(ak.prod(ak.unflatten(flat, self._nj), axis=1))

    def add_weights(self) -> None:
        """REVERTED 2026-09-15: the up_Total/down_Total swap-correction for
        WP=L0 (`_eval_flat_updown_corrected`, live 2026-09-02 through
        2026-09-15) has been removed. Component-level re-investigation found
        the individual Stat/JES/PUWeight sources that feed into Total are
        *inconsistent* for L0 (Stat cleanly normal at every pT point checked;
        JES/PUWeight mixed, sometimes normal, sometimes inverted, with no
        clean pattern) -- not the uniform signature a real swap-then-combine
        bug would produce, and not clean evidence of an intentional
        anti-correlated convention either. The "fix" was also only ever a
        magnitude-based relabeling of the file's own two reported numbers
        (whichever is larger becomes "up"), never a recomputation from the
        underlying components -- too thin a basis to keep silently altering
        the official calibration's reported uncertainty. Reverted pending
        POG/BTV confirmation of what's actually happening in this file's
        Total-combination step for L0. Official up_Total/down_Total are now
        read and used as-is, unmodified. See second-brain memory
        hczz_ctag2d_l0_updown_swap_fix (superseded) and the 2026-09-15
        revert writeup for the full before/after re-investigation.
        """
        sf_c = self._eval_event("central")
        if self._variation == "nominal":
            sf_up = self._eval_event(SYST_UP)
            sf_dn = self._eval_event(SYST_DN)
            r_up = np.where(sf_c != 0, sf_up / sf_c, 1.0)
            r_dn = np.where(sf_c != 0, sf_dn / sf_c, 1.0)
            self._weights.add(
                name=self._nuis,
                weight=sf_c,
                weightUp=sf_c * r_up,
                weightDown=sf_c * r_dn,
            )
        else:
            self._weights.add(name=self._nuis, weight=sf_c)