import numpy as np
from coffea.analysis_tools import Weights
from analysis.filesets.utils import get_nano_version
from analysis.corrections.muon import MuonWeights
from analysis.corrections.ctag import CTagCorrector
from analysis.corrections.ctag2d import CTag2DCorrector
from analysis.corrections.pileup import add_pileup_weight
from analysis.corrections.nnlops import add_nnlops_weight
from analysis.corrections.lhepdf import add_lhepdf_weight
from analysis.corrections.electron import ElectronWeights
from analysis.corrections.jerc import apply_jerc_corrections, apply_jerc_shifts
from analysis.corrections.lhescale import add_scalevar_weight
from analysis.corrections.met import apply_met_phi_corrections
from analysis.corrections.muon_ss import apply_muon_ss_corrections, apply_muon_ss_shifts
from analysis.corrections.partonshower import add_partonshower_weight
from analysis.corrections.electron_ss import apply_electron_ss_corrections
from analysis.corrections.higgs_hf import add_higgs_hf_weight
from analysis.corrections.kfactors import add_kfactor_weights


def object_corrector_manager(events, year, dataset, workflow_config):
    """apply object level corrections.

    Returns a list of (collections, shift_name) tuples -- collections is a dict of
    {"Jet"/"MET"/"Muon": <awkward array>} overrides and shift_name is a Combine-style
    shape systematic tag (or None for nominal) -- when the workflow config requests
    "jec_shifts" (JES/JER shift production, added 2026-08-18; muon scale/resolution
    shift production, CMS_scale_m/CMS_res_m, added 2026-08-19). Otherwise applies
    nominal-only corrections in place and returns None, exactly as this function
    behaved before shift support existed -- no behavior change for any workflow that
    doesn't explicitly opt in.
    """
    objcorr_config = workflow_config.corrections_config["objects"]
    if objcorr_config:
        if "jec_shifts" in objcorr_config:
            met_field_key = (
                "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"
            )
            # Apply nominal muon_ss/electon_ss corrections FIRST, before JES/JER
            # shift production -- matches this function's own non-shift ordering
            # (jec -> muon_ss -> electon_ss -> met_phi) except jec moves to last.
            # That reordering is safe: apply_jerc_shifts reads events[met_field_key]
            # fresh at call time (see get_corrected_jets_with_shifts), and
            # independent MET deltas (JEC Type-1, muon SS) commute under addition
            # -- so every JES/JER variant it produces automatically carries the
            # nominal muon/electron correction too, with zero extra plumbing.
            #
            # BUG FIXED 2026-08-19: this whole branch used to `return
            # apply_jerc_shifts(...)` immediately, silently skipping muon_ss/
            # electon_ss entirely for any workflow requesting "jec_shifts" even
            # though the yaml listed them (e.g. hplusc_mva_4class_ctag2d_
            # jecshifts.yaml) -- caught while wiring in muon shift production.
            if "muon_ss" in objcorr_config:
                apply_muon_ss_corrections(events=events, year=year)
            if "electon_ss" in objcorr_config:
                if year.startswith("202"):
                    apply_electron_ss_corrections(
                        events=events, year=year, variation="nominal"
                    )

            jec_variants = apply_jerc_shifts(events, year, dataset)
            nominal_jet = jec_variants[0][0]["Jet"]
            nominal_met = jec_variants[0][0]["MET"]
            events["Jet"] = nominal_jet
            events[met_field_key] = nominal_met

            if "muon_ss" in objcorr_config and hasattr(events, "genWeight"):
                muon_variants = apply_muon_ss_shifts(events, year, met_field_key)
            else:
                muon_variants = [
                    ({"Muon": events.Muon, met_field_key: events[met_field_key]}, None)
                ]
            nominal_muon = muon_variants[0][0]["Muon"]

            # Every returned entry explicitly carries Jet/MET/Muon, even where
            # unchanged from nominal: base.py's shift loop mutates `events`
            # cumulatively across iterations rather than resetting to nominal
            # each time, so an entry that omitted a key would silently inherit
            # whatever the PREVIOUS iteration left behind instead of nominal.
            final_variants = [
                ({"Jet": nominal_jet, met_field_key: nominal_met, "Muon": nominal_muon}, None)
            ]
            for collections, shift_name in jec_variants[1:]:
                final_variants.append(
                    (
                        {
                            "Jet": collections["Jet"],
                            met_field_key: collections["MET"],
                            "Muon": nominal_muon,
                        },
                        shift_name,
                    )
                )
            for collections, shift_name in muon_variants[1:]:
                final_variants.append(
                    (
                        {
                            "Jet": nominal_jet,
                            met_field_key: collections[met_field_key],
                            "Muon": collections["Muon"],
                        },
                        shift_name,
                    )
                )
            # Note: "met_phi" is not handled in this branch (not requested by
            # any current jec_shifts workflow) -- would need extra wiring
            # (apply per-variant, after the Muon/Jet MET is finalized) if a
            # future workflow combines jec_shifts with met_phi.
            return final_variants
        if "jec" in objcorr_config:
            # apply JEC/JER corrections
            apply_jerc_corrections(events, year, dataset)
        if "muon_ss" in objcorr_config:
            # apply muon scale and smearing corrections
            apply_muon_ss_corrections(
                events=events,
                year=year,
            )
        if "electon_ss" in objcorr_config:
            # electron scale and smearing corrections are applied only to run3 datasets
            # they are already applied in nanov9 datasets
            if year.startswith("202"):
                apply_electron_ss_corrections(
                    events=events,
                    year=year,
                    variation="nominal",
                )
        if "met_phi" in objcorr_config:
            # apply MET-phi modulation corrections
            if year.startswith("2022"):
                apply_met_phi_corrections(
                    events=events, is_mc=hasattr(events, "genWeight"), year=year
                )


def weight_manager(pruned_ev, year, dataset, workflow_config, variation="nominal"):
    """apply event level corrections (weights)"""
    # get weights config info
    weights_config = workflow_config.corrections_config["event_weights"]
    # initialize weights container
    weights_container = Weights(len(pruned_ev), storeIndividual=True)
    # get nano version
    nano_version = get_nano_version(year)
    # add weights
    if hasattr(pruned_ev, "genWeight"):
        if weights_config["genWeight"]:
            weights_container.add("genweight", pruned_ev.genWeight)

        if "pileupWeight" in weights_config:
            if weights_config["pileupWeight"]:
                add_pileup_weight(
                    events=pruned_ev,
                    year=year,
                    variation="nominal",
                    weights_container=weights_container,
                )
        if "partonshowerWeight" in weights_config:
            if weights_config["partonshowerWeight"]:
                if "PSWeight" in pruned_ev.fields:
                    add_partonshower_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                    )
        if "lhepdfWeight" in weights_config:
            if weights_config["lhepdfWeight"]:
                if "LHEPdfWeight" in pruned_ev.fields:
                    add_lhepdf_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                    )
        if "lhescaleWeight" in weights_config:
            if weights_config["lhescaleWeight"]:
                if "LHEScaleWeight" in pruned_ev.fields:
                    add_scalevar_weight(
                        events=pruned_ev, weights_container=weights_container
                    )
        if "nnlopsWeight" in weights_config:
            if weights_config["nnlopsWeight"]:
                if dataset.startswith("GluGluH"):
                    add_nnlops_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                    )
        if "kFactorWeight" in weights_config:
            if weights_config["kFactorWeight"]:
                # NNLO QCD k-factors for ggZZ (vs m4l) and NNLO QCD x NLO EW for
                # qqZZ (vs m4l, |cos theta*|), from gen-level hard-process leptons.
                # Only applied to datasets matching the ggzz/qqzz name prefixes
                # (see kfactors.py); a no-op for everything else.
                add_kfactor_weights(
                    events=pruned_ev,
                    weights_container=weights_container,
                    year=year,
                    dataset=dataset,
                    config=weights_config["kFactorWeight"],
                )
        if "higgsHFWeight" in weights_config:
            if weights_config["higgsHFWeight"]:
                # ggH/VBF only (see higgs_hf.py); per-event, keyed on gen-jet heavy
                # flavour, replaces a flat lnN on the pooled Other_Higgs group
                add_higgs_hf_weight(
                    events=pruned_ev,
                    weights_container=weights_container,
                    dataset=dataset,
                    flav="c",
                )
        if "muon" in weights_config:
            if weights_config["muon"]:
                if "selected_muons" in pruned_ev.fields:
                    muon_weights = MuonWeights(
                        events=pruned_ev,
                        year=year,
                        variation=variation,
                        weights=weights_container,
                    )
                    if "id" in weights_config["muon"]:
                        if weights_config["muon"]["id"]:
                            muon_weights.add_id_weights(
                                id_wp=weights_config["muon"]["id"]
                            )
                    if "iso" in weights_config["muon"]:
                        if weights_config["muon"]["iso"]:
                            muon_weights.add_iso_weights(
                                id_wp=weights_config["muon"]["id"],
                                iso_wp=weights_config["muon"]["iso"],
                            )
                    if "trigger" in weights_config["muon"]:
                        if weights_config["muon"]["trigger"]:
                            muon_weights.add_trigger_weights(
                                dataset=dataset,
                                id_wp=weights_config["muon"]["id"],
                                iso_wp=weights_config["muon"]["iso"],
                                hlt_paths=workflow_config.event_selection["hlt_paths"],
                            )
        if "electron" in weights_config:
            if weights_config["electron"]:
                if "selected_electrons" in pruned_ev.fields:
                    electron_weights = ElectronWeights(
                        events=pruned_ev,
                        year=year,
                        variation=variation,
                        weights=weights_container,
                    )
                    if "id" in weights_config["electron"]:
                        if weights_config["electron"]["id"]:
                            electron_weights.add_id_weights(
                                id_wp=weights_config["electron"]["id"]
                            )
                    if "reco" in weights_config["electron"]:
                        if weights_config["electron"]["reco"]:
                            if nano_version == "9":
                                electron_weights.add_reco_weights("RecoAbove20")
                                electron_weights.add_reco_weights("RecoBelow20")
                            elif nano_version == "12":
                                electron_weights.add_reco_weights("RecoBelow20")
                                electron_weights.add_reco_weights("Reco20to75")
                                electron_weights.add_reco_weights("RecoAbove75")
                            elif nano_version == "15":
                                electron_weights.add_reco_weights("Reco20to75")
                                electron_weights.add_reco_weights("RecoAbove75")
                    if "trigger" in weights_config["electron"]:
                        if weights_config["electron"]["trigger"]:
                            # no available HLT Sfs for 2024 yet
                            if nano_version != "15":
                                electron_weights.add_hlt_weights(
                                    id_wp=weights_config["electron"]["id"],
                                )

        if "ctagging" in weights_config:
            if weights_config["ctagging"]:
                working_points = ["loose", "medium", "tight"]
                if weights_config["ctagging"]["wp"] not in working_points:
                    raise ValueError(
                        f"There are no available c-tag SFs for the working point. Please specify {working_points}"
                    )
                ctag_corrector = CTagCorrector(
                    events=pruned_ev,
                    weights=weights_container,
                    worging_point=weights_config["ctagging"]["wp"],
                    year=year,
                    variation=variation,
                )
                ctag_corrector.add_ctag_weights(flavor="b")
                ctag_corrector.add_ctag_weights(flavor="c")
                ctag_corrector.add_ctag_weights(flavor="light")

        if "ctagging_2d" in weights_config:
            if weights_config["ctagging_2d"]:
                ctag2d_corrector = CTag2DCorrector(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                )
                ctag2d_corrector.add_weights()
    else:
        weights_container.add("weight", np.ones(len(pruned_ev)))
    return weights_container