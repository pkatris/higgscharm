"""
NNLO QCD (+ NLO EW) k-factors for the ZZ -> 4l backgrounds.

Port of ZZAnalysis/NanoAnalysis `weightFiller` to the coffea Weights container:

  * ggZZ (gg -> ZZ continuum): k(m4l), with PDF, alpha_S and QCD-scale variations
  * qqZZ (qq -> ZZ):           k(m4l, |cos theta*|), with QCD, EW, smoothing and
                               alpha_S variations

The generator-level m4l and cos theta* are not in standard NanoAOD (the old
module read the custom GenZZ_mass / LHEMela_costhetastar branches), so they are
computed here from the four hard-process leptons: LHEPart (status 1) when
available, otherwise GenPart with the isHardProcess flag. cos theta* is the
angle of Z1 w.r.t. the beam axis in the 4l rest frame; the k-factors only
depend on |cos theta*|, which is the same for Z1 and Z2, so the Z assignment
does not matter as long as the pairing is right (OSSF pair closest to mZ).

Expected input tables (copy them from ZZAnalysis/NanoAnalysis/data/kFactors):

  analysis/data/kfactors/gluonFusion/{Run2,Run3}-NNLO.h5
  analysis/data/kfactors/gluonFusion/{Run2,Run3}-NLO-NNLO.h5
  analysis/data/kfactors/qqBarToZZ/{Run2,Run3}/kfac_m4l__cos{0,1,2,3}.h5

Differences w.r.t. weightFiller (intentional):
  * linear interpolation with np.interp, which CLAMPS to the edge values
    outside the tabulated mass range instead of extrapolating linearly
  * fixes the qqZZ list-aliasing bug ([[None]*6]*4 made every cos bin use the
    last file) and the alpha_S variation reading the smoothing column
"""
import os
from functools import lru_cache

import numpy as np
import awkward as ak

KFACTOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kfactors"
)
MZ = 91.1876

GGZZ_COLUMNS = ["nominal", "PDF", "PDF_aS", "QCD_mu"]
QQZZ_COLUMNS = [
    "nominal",
    "QCD_up",
    "QCD_dn",
    "EW_factor",
    "smoothing_factor",
    "alpha_S_factor",
]
# |cos theta*| bin edges: [0,0.25) [0.25,0.5) [0.5,0.75) [0.75,1]
COS_EDGES = np.array([0.25, 0.5, 0.75])

# dataset-name prefixes (case-insensitive); override from the yaml if needed
DEFAULT_GGZZ_DATASETS = ["GluGluToContin"]
DEFAULT_QQZZ_DATASETS = ["ZZto4L"]


# ---------------------------------------------------------------------------
# input tables (read once per worker process)
# ---------------------------------------------------------------------------
def _run_tag(year):
    return "Run3" if str(year).startswith("202") else "Run2"


def _read_table(path, xkey, ykeys):
    import h5py

    if not os.path.isfile(path):
        raise FileNotFoundError(f"k-factor table not found: {path}")
    with h5py.File(path, "r") as f:
        x = np.asarray(f[xkey], dtype=np.float64).ravel()
        order = np.argsort(x)  # np.interp needs increasing x
        ys = {
            key: np.asarray(f[key], dtype=np.float64).ravel()[order] for key in ykeys
        }
    return x[order], ys


@lru_cache(maxsize=None)
def _ggzz_table(run, order):
    fname = {1: "NNLO", 2: "NLO-NNLO"}[order]
    path = os.path.join(KFACTOR_DIR, "gluonFusion", f"{run}_K-{fname}.h5")
    return _read_table(path, "hmass", GGZZ_COLUMNS)


@lru_cache(maxsize=None)
def _qqzz_tables(run):
    # bin i covers |cos theta*| in [0.25*i, 0.25*(i+1)). The files are indexed on
    # the negative side (cos0 = [-1,-0.75] ... cos3 = [-0.25,0]), hence 3-i.
    return [
        _read_table(
            os.path.join(KFACTOR_DIR, "qqBarToZZ", run, f"kfac_m4l__cos{3 - i}.h5"),
            "zzmass",
            QQZZ_COLUMNS,
        )
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# generator-level ZZ kinematics
# ---------------------------------------------------------------------------
def _hard_leptons(events):
    if "LHEPart" in events.fields:
        parts = events.LHEPart
        sel = parts.status == 1
    else:
        parts = events.GenPart
        sel = parts.hasFlags(["isHardProcess"])
    apdg = abs(parts.pdgId)
    sel = sel & ((apdg == 11) | (apdg == 13) | (apdg == 15))
    return parts[sel]


def gen_zz_kinematics(events):
    """
    Returns (m4l, |cos theta*|) as numpy arrays of len(events).
    Both are NaN for events without exactly four hard-process leptons
    (and |cos theta*| is NaN if no OSSF pair is found).
    """
    n = len(events)
    m4l = np.full(n, np.nan)
    abscos = np.full(n, np.nan)

    leps = _hard_leptons(events)
    has4 = ak.to_numpy(ak.num(leps) == 4)
    if not np.any(has4):
        return m4l, abscos
    leps = leps[has4]

    def to_np(x):
        return ak.to_numpy(ak.flatten(x)).reshape(-1, 4).astype(np.float64)

    pt, eta, phi, mass = to_np(leps.pt), to_np(leps.eta), to_np(leps.phi), to_np(leps.mass)
    pdg = ak.to_numpy(ak.flatten(leps.pdgId)).reshape(-1, 4)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    p4 = np.stack([e, px, py, pz], axis=-1)  # (nev, 4 leptons, [E,px,py,pz])

    def inv_mass(v):
        return np.sqrt(np.maximum(v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, axis=-1), 0.0))

    zz = p4.sum(axis=1)
    m = inv_mass(zz)

    # Z1 = OSSF pair closest to mZ (its complement is then OSSF as well)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    nev = len(zz)
    zcand = np.empty((nev, len(pairs), 4))
    dist = np.full((nev, len(pairs)), np.inf)
    for k, (i, j) in enumerate(pairs):
        zcand[:, k] = p4[:, i] + p4[:, j]
        ossf = pdg[:, i] == -pdg[:, j]
        dist[:, k] = np.where(ossf, np.abs(inv_mass(zcand[:, k]) - MZ), np.inf)
    rows = np.arange(nev)
    best = np.argmin(dist, axis=1)
    found = np.isfinite(dist[rows, best])
    z1 = zcand[rows, best]

    # boost Z1 into the 4l rest frame
    beta = zz[:, 1:] / zz[:, [0]]
    b2 = np.clip(np.sum(beta**2, axis=1), 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * z1[:, 1:], axis=1)
    g2 = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    p_star = z1[:, 1:] + ((g2 * bp - gamma * z1[:, 0])[:, None]) * beta
    norm = np.linalg.norm(p_star, axis=1)
    cos = np.divide(p_star[:, 2], norm, out=np.zeros_like(norm), where=norm > 0)

    m4l[has4] = m
    abscos[has4] = np.where(found, np.abs(cos), np.nan)
    return m4l, abscos


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def _interp(x, xs, ys, fill):
    valid = np.isfinite(x)
    out = np.full(len(x), fill, dtype=np.float64)
    out[valid] = np.interp(x[valid], xs, ys)
    return out


def add_ggzz_kfactor(events, weights_container, year, order=1):
    """
    order = 1: LO -> NNLO, order = 2: NLO -> NNLO.
    Adds 'kfactor_ggzz' with variations kfactor_ggzz_{pdf,as,qcd}{Up,Down}.
    Variations are relative: k * (1 +- factor).
    """
    if order not in (1, 2):
        raise ValueError(
            f"ggzz k-factor order must be 1 (LO->NNLO) or 2 (NLO->NNLO), got {order}"
        )
    xs, ys = _ggzz_table(_run_tag(year), order)
    m4l, _ = gen_zz_kinematics(events)

    k = _interp(m4l, xs, ys["nominal"], 1.0)
    rel = {
        "pdf": _interp(m4l, xs, ys["PDF"], 0.0),
        "as": _interp(m4l, xs, ys["PDF_aS"], 0.0),
        "qcd": _interp(m4l, xs, ys["QCD_mu"], 0.0),
    }
    weights_container.add_multivariation(
        name="kfactor_ggzz",
        weight=k,
        modifierNames=list(rel),
        weightsUp=[k * (1.0 + f) for f in rel.values()],
        weightsDown=[k * (1.0 - f) for f in rel.values()],
    )


def add_qqzz_kfactor(events, weights_container, year):
    """
    Adds 'kfactor_qqzz' with variations kfactor_qqzz_{qcd,ew,smoothing,as}{Up,Down}.
    QCD up/down are multiplicative factors on top of the nominal k-factor;
    EW, smoothing and alpha_S are relative: k * (1 +- factor).
    """
    tables = _qqzz_tables(_run_tag(year))
    m4l, abscos = gen_zz_kinematics(events)
    valid = np.isfinite(m4l) & np.isfinite(abscos)
    ibin = np.digitize(np.nan_to_num(abscos), COS_EDGES)  # 0..3

    vals = {
        col: np.full(len(m4l), 1.0 if col in ("nominal", "QCD_up", "QCD_dn") else 0.0)
        for col in QQZZ_COLUMNS
    }
    for i, (xs, ys) in enumerate(tables):
        sel = valid & (ibin == i)
        if np.any(sel):
            for col in QQZZ_COLUMNS:
                vals[col][sel] = np.interp(m4l[sel], xs, ys[col])

    k = vals["nominal"]
    weights_container.add_multivariation(
        name="kfactor_qqzz",
        weight=k,
        modifierNames=["qcd", "ew", "smoothing", "as"],
        weightsUp=[
            k * vals["QCD_up"],
            k * (1.0 + vals["EW_factor"]),
            k * (1.0 + vals["smoothing_factor"]),
            k * (1.0 + vals["alpha_S_factor"]),
        ],
        weightsDown=[
            k * vals["QCD_dn"],
            k * (1.0 - vals["EW_factor"]),
            k * (1.0 - vals["smoothing_factor"]),
            k * (1.0 - vals["alpha_S_factor"]),
        ],
    )


def _normalize_config(config):
    if config is True:
        return {"ggzz": 1, "qqzz": True}
    if isinstance(config, list):  # yaml list of single-key dicts
        merged = {}
        for item in config:
            merged.update(item)
        return merged
    return dict(config) if config else {}


def _matches(dataset, prefixes):
    return any(dataset.lower().startswith(p.lower()) for p in prefixes)


def add_kfactor_weights(events, weights_container, year, dataset, config):
    """
    Dispatcher used by weight_manager. `config` is the yaml `kFactorWeight` entry:

        kFactorWeight:
          ggzz: 1          # 0 = off, 1 = LO->NNLO, 2 = NLO->NNLO
          qqzz: true
          ggzz_datasets: [GluGluToContin]   # optional, dataset-name prefixes
          qqzz_datasets: [ZZto4L]           # optional
    """
    config = _normalize_config(config)
    ggzz_order = int(config.get("ggzz", 0) or 0)
    apply_qqzz = bool(config.get("qqzz", False))
    ggzz_datasets = config.get("ggzz_datasets", DEFAULT_GGZZ_DATASETS)
    qqzz_datasets = config.get("qqzz_datasets", DEFAULT_QQZZ_DATASETS)

    if ggzz_order and _matches(dataset, ggzz_datasets):
        add_ggzz_kfactor(events, weights_container, year, order=ggzz_order)
    if apply_qqzz and _matches(dataset, qqzz_datasets):
        add_qqzz_kfactor(events, weights_container, year)