# tools to apply JEC/JER and compute their uncertainties
# JME recommendations: https://cms-jerc.web.cern.ch/Recommendations/
# copied from https://github.com/green-cabbage/copperheadV2/blob/main/corrections/jet.py and https://github.com/cms-btv-pog/BTVNanoCommissioning/blob/master/src/BTVNanoCommissioning/utils/correction.py
import copy
import yaml
import warnings
import contextlib
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from analysis.corrections.utils import correction_files
from analysis.filesets.utils import get_dataset_era, get_nano_version
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools.CorrectedMETFactory import corrected_polar_met


def update_met(events, year):
    # Type-I MET correction, from corrected MET factory
    # https://github.com/scikit-hep/coffea/blob/master/src/coffea/jetmet_tools/CorrectedMETFactory.py
    met_field_key = "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"
    nocorrmet = events[met_field_key]
    jets = events.Jet

    met_inputs = [nocorrmet.pt, nocorrmet.phi, jets.pt, jets.phi, jets.pt_raw]

    met = copy.deepcopy(nocorrmet)
    met["pt"], met["phi"] = (
        ak.values_astype(corrected_polar_met(*met_inputs).pt, np.float32),
        ak.values_astype(corrected_polar_met(*met_inputs).phi, np.float32),
    )
    met["orig_pt"], met["orig_phi"] = nocorrmet["pt"], nocorrmet["phi"]

    # update MET collection
    events[met_field_key] = met


# ----------------------------------------------------------------------------------------------------
# COFFEA'S JET/MET TOOLS
# ----------------------------------------------------------------------------------------------------
with importlib.resources.open_text(
    f"analysis.corrections", f"jerc_params.yaml"
) as file:
    JEC_PARAMS = yaml.safe_load(file)

with importlib.resources.open_text(
    f"analysis.corrections", f"jec_params_correctionlib.yaml"
) as file:
    JEC_PARAMS_CORRECTIONLIB = yaml.safe_load(file)


def jec_names_and_sources(year):
    names = {}
    algo = JEC_PARAMS["algorithm"][year]
    suffix = {
        "jec_names": [
            f"_{level}_{algo}" for level in JEC_PARAMS["jec_levels_mc"][year]
        ],
        "jec_names_data": [
            f"_{level}_{algo}" for level in JEC_PARAMS["jec_levels_data"][year]
        ],
        "junc_names": [f"_Uncertainty_{algo}"],
        "junc_names_data": [f"_Uncertainty_{algo}"],
        "junc_sources": [f"_UncertaintySources_{algo}"],
        "junc_sources_data": [f"_UncertaintySources_{algo}"],
        "jer_names": [f"_PtResolution_{algo}"],
        "jersf_names": [f"_SF_{algo}"],
    }
    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in JEC_PARAMS["runs"][year]:
                for tag, iruns in JEC_PARAMS["jec_data_tags"][year].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = (
                JEC_PARAMS["jer_tags"][year]
                if "jer" in key
                else JEC_PARAMS["jec_tags"][year]
            )
            names[key] = [f"{tag}{s}" for s in suff]
    return names


def get_jet_evaluator(year):
    names = jec_names_and_sources(year)
    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }
    # prepare evaluators for JEC, JER and their systematics
    jec_ext = extractor()
    for opt, ext in extensions.items():
        # MC
        with contextlib.ExitStack() as stack:
            jec_files = [
                stack.enter_context(
                    importlib.resources.path("analysis.data.jec", f"{name}.{ext}.txt")
                )
                for name in names[opt]
            ]
            jec_ext.add_weight_sets([f"* * {file}" for file in jec_files])
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        with contextlib.ExitStack() as stack:
            jec_data_files = [
                stack.enter_context(
                    importlib.resources.path("analysis.data.jec", f"{name}.{ext}.txt")
                )
                for name in data
            ]
            jec_ext.add_weight_sets([f"* * {file}" for file in jec_data_files])

    jec_ext.finalize()
    jet_evaluator = jec_ext.make_evaluator()
    return jet_evaluator


def apply_jerc_coffea(
    events,
    year,
    dataset,
):
    is_mc = hasattr(events, "genWeight")

    # add requiered variables to Jet collection
    jets = events.Jet

    # set raw pT and Mass, otherwise original pT and Mass will be used as 'raw' values
    events["Jet", "pt_raw"] = (1 - jets.rawFactor) * jets.pt
    events["Jet", "mass_raw"] = (1 - jets.rawFactor) * jets.mass
    if is_mc:
        # set ptGenJet (required for hybrid JER smearing method)
        events["Jet", "pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
    events["Jet", "rho"] = (
        ak.ones_like(jets.pt) * events.Rho.fixedGridRhoFastjetAll
        if hasattr(events, "Rho")
        else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    )

    # set inputs for jec, jer and junc stack
    names = jec_names_and_sources(year)
    jet_evaluator = get_jet_evaluator(year)

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }
    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])
    jec_input_options = {}
    jet_variations = ["jec", "jer", "junc"]
    for variation in jet_variations:
        jec_input_options[variation] = {}
        for name in stacks[f"{variation}_stack"]:
            jec_input_options[variation][name] = jet_evaluator[name]
    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    jec_options = {}
    jec_options.update(jec_input_options["jec"])
    jec_options.update(jec_input_options["junc"])
    if is_mc:
        jec_options.update(jec_input_options["jer"])

    # set jerc name map (I don't use JECStack.blank_name_map since it includes 'ptRaw' and 'massRaw' by default)
    jec_name_map = {
        "JetPt": "pt",
        "JetMass": "mass",
        "JetEta": "eta",
        "JetA": "area",
        "ptGenJet": "pt_gen",
        "Rho": "rho",
        "METpt": "pt",
        "METphi": "phi",
        "JetPhi": "phi",
        "UnClusteredEnergyDeltaX": (
            "MetUnclustEnUpDeltaX" if hasattr(events, "Rho") else None
        ),
        "UnClusteredEnergyDeltaY": (
            "MetUnclustEnUpDeltaY" if hasattr(events, "Rho") else None
        ),
    }
    jec_name_map.update(
        {
            "ptRaw": "pt_raw",
            "massRaw": "mass_raw",
        }
    )

    if is_mc:
        # create MC factory with jec, jer and junc stack
        stack = JECStack(jec_options)
        jec_factory = CorrectedJetsFactory(jec_name_map, stack)
    else:
        # create a separate factory for the data era
        era = get_dataset_era(dataset, year)
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][era]}
            )
        for src in names["junc_sources_data"][era]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]
        jec_stack_data = JECStack(jec_inputs_data)
        jec_factory = CorrectedJetsFactory(jec_name_map, jec_stack_data)

    # update Jet collection
    events["Jet"] = jec_factory.build(events.Jet, events.caches[0])
    # update MET collection
    update_met(events, year)
    # TO DO: SYSTEMATICS


# ----------------------------------------------------------------------------------------------------
# CORRECTIONLIB
# ----------------------------------------------------------------------------------------------------

# FIXED 2026-10-21: single source of truth for correctionlib JEC/JER tags, shared by the
# nominal (apply_jerc_correctionlib) and shift (get_corrected_jets_with_shifts) paths.
# Previously the nominal path hardcoded 2024 = Summer24Prompt24_V1 while the shift path read
# jec_params_correctionlib.yaml (2024 = Summer24Prompt24_V2, not in the pinned 2024 JSON ->
# KeyError 'Summer24Prompt24_V2_MC_L1L2L3Res_AK4PFPuppi'), and the shift path looked up the
# data tag by era (params['C'] -> KeyError) although 2024 has a single "Data" tag. Entries
# here take precedence over the yaml so nominal and shifted jets always use the same JEC.
CORRECTIONLIB_TAG_OVERRIDES = {
    "2024": {
        # For the time being, use the Summer23BPix JERs for 2024 data (November 2025)
        "MC": "Summer24Prompt24_V1 Summer23BPixPrompt23_RunD_JRV1",
        "Data": "Summer24Prompt24_V1",
    }
}

JEC_ALGO_CORRECTIONLIB = "AK4PFPuppi"


def get_correctionlib_tags(year):
    """{'MC': '<jec_tag> <jer_tag>', 'Data': '<jec_tag>' and/or '<era>': '<jec_tag>'}"""
    if year in CORRECTIONLIB_TAG_OVERRIDES:
        return CORRECTIONLIB_TAG_OVERRIDES[year]
    return JEC_PARAMS_CORRECTIONLIB[year]


def get_correctionlib_data_tag(params, dataset, year):
    """Per-era data tag if the year defines one (2022/2023), else the single 'Data' tag (2024)."""
    era = get_dataset_era(dataset, year)
    tag = params.get(era, params.get("Data"))
    if tag is None:
        raise KeyError(
            f"No JEC data tag for year '{year}', era '{era}' "
            f"(available keys: {list(params)})"
        )
    return tag


def check_correction_names(cset, names, year):
    """Fail early with a readable message if a correction is missing from the JERC JSON."""
    available = set(cset.keys()) | set(cset.compound.keys())
    missing = [n for n in names if n not in available]
    if missing:
        tags = sorted(
            {k.rsplit(f"_{JEC_ALGO_CORRECTIONLIB}", 1)[0].rsplit("_", 1)[0] for k in available}
        )
        raise KeyError(
            f"JERC JSON for '{year}' ({correction_files['jerc'][year]}) has no {missing}. "
            f"Tags present in the file: {tags}"
        )


def load_jersmear():
    cset_jersmear_paths = [
        "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/JER-Smearing/latest/jer_smear.json.gz",
        "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/jer_smear.json.gz",
    ]
    for _jersmear_path in cset_jersmear_paths:
        if Path(_jersmear_path).exists():
            return correctionlib.CorrectionSet.from_file(_jersmear_path)["JERSmear"]
    warnings.warn(
        "JER smearing JSON not found in CVMFS. JER smearing corrections will be disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


# from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py
def get_corr_inputs(input_dict, corr_obj, jersyst="nom"):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = []
    for inputs in corr_obj.inputs:
        if "systematic" in inputs.name:
            input_values.append(jersyst)
        else:
            input_values.append(
                np.array(
                    input_dict[
                        inputs.name.replace("Jet", "")
                        .replace("Pt", "pt")
                        .replace("Phi", "phi")
                        .replace("Eta", "eta")
                        .replace("Mass", "mass")
                        .replace("Rho", "rho")
                        .replace("A", "area")
                    ]
                )
            )
    return input_values


def apply_jerc_correctionlib(events, year, dataset):
    params = get_correctionlib_tags(year)
    sf_jersmear = load_jersmear()

    cset = correctionlib.CorrectionSet.from_file(correction_files["jerc"][year])

    is_mc = hasattr(events, "genWeight")
    if is_mc:
        jecname, jrname = params["MC"].split(" ")
        jecname += "_MC"
        jrname += "_MC"
        check_correction_names(
            cset,
            [
                f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}",
                f"{jrname}_ScaleFactor_{JEC_ALGO_CORRECTIONLIB}",
                f"{jrname}_PtResolution_{JEC_ALGO_CORRECTIONLIB}",
            ],
            year,
        )
    else:
        jecname = get_correctionlib_data_tag(params, dataset, year) + "_DATA"
        check_correction_names(
            cset, [f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}"], year
        )

    # store the original jet info
    nocorrjet = events.Jet
    nocorrjet["pt_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["pt"]
    nocorrjet["mass_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["mass"]
    nocorrjet["rho"] = (
        ak.ones_like(nocorrjet.pt) * events.Rho.fixedGridRhoFastjetAll
        if hasattr(events, "Rho")
        else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, nocorrjet.pt)[0]
    )
    nocorrjet["EventID"] = ak.broadcast_arrays(events.event, nocorrjet.pt)[0]
    nocorrjet["run"] = ak.broadcast_arrays(events.run, nocorrjet.pt)[0]
    if is_mc:
        genjetidx = ak.where(nocorrjet.genJetIdx == -1, 0, nocorrjet.genJetIdx)
        nocorrjet["Genpt"] = ak.where(
            nocorrjet.genJetIdx == -1, -1, events.GenJet[genjetidx].pt
        )
    jets = copy.deepcopy(nocorrjet)
    jets["orig_pt"] = ak.values_astype(nocorrjet["pt"], np.float32)

    # flatten jets
    j, nj = ak.flatten(nocorrjet), ak.num(nocorrjet)

    # JEC
    jec_corr = cset.compound[f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}"]
    jec_input = get_corr_inputs(j, jec_corr)
    jec_flat_corr_factor = jec_corr.evaluate(*jec_input)

    ## JER
    if is_mc:
        jer_sf = cset[f"{jrname}_ScaleFactor_{JEC_ALGO_CORRECTIONLIB}"]
        jer_ptres = cset[f"{jrname}_PtResolution_{JEC_ALGO_CORRECTIONLIB}"]
        # for MC, correct the jet pT with JEC first
        j["pt"] = j["pt_raw"] * jec_flat_corr_factor
        j["mass"] = j["mass_raw"] * jec_flat_corr_factor
        jer_sf_input = get_corr_inputs(j, jer_sf)
        jer_ptres_input = get_corr_inputs(j, jer_ptres)
        j["JER"] = jer_ptres.evaluate(*jer_ptres_input)
        j["JERSF"] = jer_sf.evaluate(*jer_sf_input)
        if sf_jersmear is None:
            warnings.warn(
                "JER smearing coefficients unavailable. "
                "Proceeding without applying JER smearing.",
                RuntimeWarning,
                stacklevel=2,
            )
            corr_factor = jec_flat_corr_factor
        else:
            jer_smear_input = get_corr_inputs(j, sf_jersmear)
            corr_factor = jec_flat_corr_factor * sf_jersmear.evaluate(*jer_smear_input)
    else:
        # in data only the JEC is applied
        corr_factor = jec_flat_corr_factor

    corr_factor = ak.unflatten(corr_factor, nj)

    jets["pt"] = ak.values_astype(nocorrjet["pt_raw"] * corr_factor, np.float32)
    jets["mass"] = ak.values_astype(nocorrjet["mass_raw"] * corr_factor, np.float32)

    # update Jet collection
    events["Jet"] = jets
    # update MET collection
    update_met(events, year)
    # TO DO: SYSTEMATICS


def apply_jerc_corrections(events, year, dataset):
    if year == "2024":
        apply_jerc_correctionlib(events, year, dataset)
    else:
        apply_jerc_coffea(events, year, dataset)


# ----------------------------------------------------------------------------------------------------
# JES/JER SHIFT PRODUCTION (added 2026-08-18)
# Additive only -- does not touch apply_jerc_corrections/apply_jerc_coffea/apply_jerc_correctionlib
# above, so every existing caller keeps today's exact single-pass, nominal-only behavior unless it
# explicitly opts into apply_jerc_shifts below. Scope: Total JES + JER only (2 nuisances,
# CMS_scale_j_<year>/CMS_res_j_<year>), matching the simpler of two schemes considered (the
# alternative -- an 11-source "Regrouped" JES split matching Felix Heyen's thesis -- needs a
# Regrouped JEC uncertainty file this repo doesn't have; deferred, not started).
# ----------------------------------------------------------------------------------------------------


def get_corrected_jets_with_shifts(events, year):
    """Build corrected Jet/MET collections plus .JES_jes.up/.down and .JER.up/.down
    variant sub-collections, using the correctionlib JERC json (same file/tags as
    apply_jerc_correctionlib, extended here to also evaluate the Total JES
    uncertainty and the JER SF's up/down systematic). MC only -- data has no JES/JER
    variations to produce.
    """
    is_mc = hasattr(events, "genWeight")
    dataset = events.metadata["dataset"]

    sf_jersmear = load_jersmear()

    cset = correctionlib.CorrectionSet.from_file(correction_files["jerc"][year])
    params = get_correctionlib_tags(year)
    if is_mc:
        jecname, jrname = params["MC"].split(" ")
        jecname += "_MC"
        jrname += "_MC"
        check_correction_names(
            cset,
            [
                f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}",
                f"{jecname}_Total_{JEC_ALGO_CORRECTIONLIB}",
                f"{jrname}_ScaleFactor_{JEC_ALGO_CORRECTIONLIB}",
                f"{jrname}_PtResolution_{JEC_ALGO_CORRECTIONLIB}",
            ],
            year,
        )
    else:
        jecname = get_correctionlib_data_tag(params, dataset, year) + "_DATA"
        check_correction_names(
            cset, [f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}"], year
        )

    nocorrjet = events.Jet
    nocorrjet["pt_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["pt"]
    nocorrjet["mass_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["mass"]
    nocorrjet["rho"] = (
        ak.ones_like(nocorrjet.pt) * events.Rho.fixedGridRhoFastjetAll
        if hasattr(events, "Rho")
        else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, nocorrjet.pt)[0]
    )
    # required by the JER-smearing correction's "hybrid" method (matched-gen-jet pt
    # when available else deterministic stochastic smearing seeded from EventID/run)
    # -- missing these fields was the bug caught by the 2026-08-18 smoke test
    # (ValueError: key "Genpt"/"EventID" does not exist), fixed on the second and
    # third iterations respectively.
    nocorrjet["EventID"] = ak.broadcast_arrays(events.event, nocorrjet.pt)[0]
    nocorrjet["run"] = ak.broadcast_arrays(events.run, nocorrjet.pt)[0]
    if is_mc:
        genjetidx = ak.where(nocorrjet.genJetIdx == -1, 0, nocorrjet.genJetIdx)
        nocorrjet["Genpt"] = ak.where(
            nocorrjet.genJetIdx == -1, -1, events.GenJet[genjetidx].pt
        )
    jets = copy.deepcopy(nocorrjet)
    j, nj = ak.flatten(nocorrjet), ak.num(nocorrjet)

    jec_corr = cset.compound[f"{jecname}_L1L2L3Res_{JEC_ALGO_CORRECTIONLIB}"]
    jec_flat_corr_factor = jec_corr.evaluate(*get_corr_inputs(j, jec_corr))

    if is_mc:
        jer_sf = cset[f"{jrname}_ScaleFactor_{JEC_ALGO_CORRECTIONLIB}"]
        jer_ptres = cset[f"{jrname}_PtResolution_{JEC_ALGO_CORRECTIONLIB}"]
        j["pt"] = j["pt_raw"] * jec_flat_corr_factor
        j["mass"] = j["mass_raw"] * jec_flat_corr_factor
        j["JER"] = jer_ptres.evaluate(*get_corr_inputs(j, jer_ptres))
        j["JERSF"] = jer_sf.evaluate(*get_corr_inputs(j, jer_sf))
        if sf_jersmear is None:
            corr_factor = jec_flat_corr_factor
        else:
            corr_factor = jec_flat_corr_factor * sf_jersmear.evaluate(
                *get_corr_inputs(j, sf_jersmear)
            )
    else:
        corr_factor = jec_flat_corr_factor
    corr_factor = ak.unflatten(corr_factor, nj)

    jets["pt"] = ak.values_astype(nocorrjet["pt_raw"] * corr_factor, np.float32)
    jets["mass"] = ak.values_astype(nocorrjet["mass_raw"] * corr_factor, np.float32)

    met_field_key = "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"
    nocorrmet = events[met_field_key]
    met_inputs = [nocorrmet.pt, nocorrmet.phi, jets.pt, jets.phi, jets.pt_raw]
    met = copy.deepcopy(nocorrmet)
    met["pt"], met["phi"] = (
        ak.values_astype(corrected_polar_met(*met_inputs).pt, np.float32),
        ak.values_astype(corrected_polar_met(*met_inputs).phi, np.float32),
    )

    if is_mc:
        jesuncmap = cset[f"{jecname}_Total_{JEC_ALGO_CORRECTIONLIB}"]
        jesunc = ak.unflatten(jesuncmap.evaluate(j.eta, j.pt), nj)
        unc_jets, unc_met = {}, {}
        for var in ["up", "down"]:
            fac = 1.0 if var == "up" else -1.0
            # JES total
            unc_jets[f"jes{var}"] = copy.copy(jets)
            unc_jets[f"jes{var}"]["pt"] = ak.values_astype(
                jets["pt"] * (1 + fac * jesunc), np.float32
            )
            unc_jets[f"jes{var}"]["mass"] = ak.values_astype(
                jets["mass"] * (1 + fac * jesunc), np.float32
            )
            unc_met_inputs = [
                nocorrmet.pt, nocorrmet.phi,
                unc_jets[f"jes{var}"]["pt"], jets.phi, jets.pt_raw,
            ]
            unc_met[f"jes{var}"] = copy.copy(met)
            unc_met[f"jes{var}"]["pt"] = ak.values_astype(
                corrected_polar_met(*unc_met_inputs).pt, np.float32
            )
            unc_met[f"jes{var}"]["phi"] = ak.values_astype(
                corrected_polar_met(*unc_met_inputs).phi, np.float32
            )

            # JER (SF evaluated with the up/down systematic, same smearing formula)
            if sf_jersmear is None:
                unc_jets[f"jer{var}"] = copy.copy(jets)
                unc_met[f"jer{var}"] = copy.copy(met)
            else:
                j["JERSF"] = jer_sf.evaluate(*get_corr_inputs(j, jer_sf, jersyst=var))
                corr_factor_var = ak.unflatten(
                    jec_flat_corr_factor * sf_jersmear.evaluate(*get_corr_inputs(j, sf_jersmear)),
                    nj,
                )
                unc_jets[f"jer{var}"] = copy.copy(nocorrjet)
                unc_jets[f"jer{var}"]["pt"] = ak.values_astype(
                    nocorrjet["pt_raw"] * corr_factor_var, np.float32
                )
                unc_jets[f"jer{var}"]["mass"] = ak.values_astype(
                    nocorrjet["mass_raw"] * corr_factor_var, np.float32
                )
                unc_jer_met_inputs = [
                    nocorrmet.pt, nocorrmet.phi,
                    unc_jets[f"jer{var}"]["pt"], jets.phi, jets.pt_raw,
                ]
                unc_met[f"jer{var}"] = copy.copy(met)
                unc_met[f"jer{var}"]["pt"] = ak.values_astype(
                    corrected_polar_met(*unc_jer_met_inputs).pt, np.float32
                )
                unc_met[f"jer{var}"]["phi"] = ak.values_astype(
                    corrected_polar_met(*unc_jer_met_inputs).phi, np.float32
                )

        jets["JES_jes"] = ak.zip({"up": unc_jets["jesup"], "down": unc_jets["jesdown"]})
        jets["JER"] = ak.zip({"up": unc_jets["jerup"], "down": unc_jets["jerdown"]})
        met["JES_jes"] = ak.zip({"up": unc_met["jesup"], "down": unc_met["jesdown"]})
        met["JER"] = ak.zip({"up": unc_met["jerup"], "down": unc_met["jerdown"]})

    return jets, met


def apply_jerc_shifts(events, year, dataset):
    """Return a list of (collections_dict, shift_name) tuples: nominal plus, for MC
    only, CMS_scale_j_<year>Up/Down and CMS_res_j_<year>Up/Down. Mirrors
    apply_jerc_correctionlib's era coverage (correctionlib-based eras only --
    2022preEE/2022postEE/2023preBPix/2023postBPix/2024; the coffea/.txt-file eras
    have no shift production wired, same as before this addition).
    """
    if year not in correction_files["jerc"]:
        raise ValueError(
            f"apply_jerc_shifts has no correctionlib JERC file for year '{year}' "
            f"(only {list(correction_files['jerc'])} are wired)."
        )
    jets, met = get_corrected_jets_with_shifts(events, year)
    shifts = [({"Jet": jets, "MET": met}, None)]
    if hasattr(events, "genWeight"):
        y4 = year[:4]
        shifts += [
            ({"Jet": jets.JES_jes.up, "MET": met.JES_jes.up}, f"CMS_scale_j_{y4}Up"),
            ({"Jet": jets.JES_jes.down, "MET": met.JES_jes.down}, f"CMS_scale_j_{y4}Down"),
            ({"Jet": jets.JER.up, "MET": met.JER.up}, f"CMS_res_j_{y4}Up"),
            ({"Jet": jets.JER.down, "MET": met.JER.down}, f"CMS_res_j_{y4}Down"),
        ]
    return shifts