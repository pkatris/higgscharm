import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods.vector import LorentzVector
from analysis.working_points import working_points
from analysis.selections import (
    delta_r_higher,
    delta_r_lower,
    select_dileptons,
    transverse_mass,
    get_closest_lepton,
    assign_lepton_fsr_idx,
    fourlepcand,
    make_cand,
    select_best_zzcandidate,
    select_candidate_mass,
    select_candidate_cjet_dphi,
)


class ObjectSelector:

    def __init__(self, object_selection_config, year):
        self.object_selection_config = object_selection_config
        self.year = year

    def select_objects(self, events):
        self.objects = {}
        self.events = events

        for obj_name, obj_config in self.object_selection_config.items():
            # check if object is defined from events or user defined function
            if "events" in obj_config["field"]:
                self.objects[obj_name] = eval(obj_config["field"])
            else:
                selection_function = getattr(self, obj_config["field"])
                selection_function(obj_name)
            if "add_cut" in obj_config:
                for field_to_add in obj_config["add_cut"]:
                    selection_mask = self.get_selection_mask(
                        events=events,
                        obj_name=obj_name,
                        cuts=obj_config["add_cut"][field_to_add],
                    )
                    self.objects[obj_name][field_to_add] = selection_mask
            if "add_field" in obj_config:
                for field_name, field_to_add in obj_config["add_field"].items():
                    self.objects[obj_name][field_name] = eval(field_to_add)
            if "cuts" in obj_config:
                selection_mask = self.get_selection_mask(
                    events=events, obj_name=obj_name, cuts=obj_config["cuts"]
                )
                self.objects[obj_name] = self.objects[obj_name][selection_mask]
        return self.objects

    def get_selection_mask(self, events, obj_name, cuts):
        # bring objects and year to local scope
        objects = self.objects
        year = self.year
        # initialize selection mask
        selection_mask = ak.ones_like(self.objects[obj_name].pt, dtype=bool)
        # iterate over all cuts
        for str_mask in cuts:
            mask = eval(str_mask)
            selection_mask = np.logical_and(selection_mask, mask)
        return selection_mask

    # --------------------------------------------------------------------------------
    # ZToLL
    # --------------------------------------------------------------------------------
    def select_dimuons(self, obj_name):
        if "muons" not in self.objects:
            raise ValueError(f"'muons' object has not been defined!")
        self.objects[obj_name] = select_dileptons(self.objects, "muons")

    def select_dielectrons(self, obj_name):
        if "electrons" not in self.objects:
            raise ValueError(f"'electrons' object has not been defined!")
        self.objects[obj_name] = select_dileptons(self.objects, "electrons")

    # --------------------------------------------------------------------------------
    # ZZTo4L (Z+L, ZLL)
    # --------------------------------------------------------------------------------
    def select_zzto4l_leptons(self, obj_name):
        muons = self.objects["muons"]
        muons["lostHits"] = ak.zeros_like(muons.pt)
        # Add placeholder mvaHZZIso for muons (electrons use this for tight ID)
        muons["mvaHZZIso"] = ak.ones_like(muons.pt) * -999
        # muons already have isPFcand and highPtId from NanoAOD
        electrons = self.objects["electrons"]
        # electrons already have mvaHZZIso from NanoAOD
        # Add placeholder isPFcand and highPtId for electrons (muons use these for tight ID)
        electrons["isPFcand"] = ak.ones_like(electrons.pt, dtype=bool)
        electrons["highPtId"] = ak.zeros_like(electrons.pt, dtype=int)
        # leptons before FSR recovery/iso correction
        helper_leptons = ak.concatenate([muons, electrons], axis=1)
        helper_leptons["idx"] = ak.local_index(helper_leptons, axis=1)
        helper_leptons = ak.zip(
            {
                "pt": helper_leptons.pt,
                "eta": helper_leptons.eta,
                "phi": helper_leptons.phi,
                "mass": helper_leptons.mass,
                "charge": helper_leptons.charge,
                "pdgId": helper_leptons.pdgId,
                "idx": helper_leptons.idx,
                "is_loose": helper_leptons.is_loose,
                "is_relaxed": helper_leptons.is_relaxed,
                "is_tight": helper_leptons.is_tight,
                "lostHits": helper_leptons.lostHits,
                "sip3d": helper_leptons.sip3d,
                "pfRelIso03_all": helper_leptons.pfRelIso03_all,
                "mvaHZZIso": helper_leptons.mvaHZZIso,
                "isPFcand": helper_leptons.isPFcand,
                "highPtId": helper_leptons.highPtId,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        # select FSR photons
        fsr_photons = self.objects["fsr_photons"]
        fsr_photons["mass"] = 0
        fsr_photons["charge"] = 0
        fsr_photons["idx"] = ak.local_index(fsr_photons, axis=1)
        # find closest lepton
        closest_lepton, dr_et2 = get_closest_lepton(fsr_photons, helper_leptons)
        fsr_photons["helper_idx"] = ak.local_index(closest_lepton, axis=1)
        fsr_photons["lepton_idx"] = ak.fill_none(closest_lepton.idx, -1)
        # if the FSR photon is matched to multiple leptons, choose the one with lowest dREt2
        fsr_cartesian = ak.cartesian(
            {"fsr1": fsr_photons, "fsr2": fsr_photons}, nested=True, axis=1
        )
        is_duplicate = (
            ak.sum(
                fsr_cartesian.fsr1.lepton_idx == fsr_cartesian.fsr2.lepton_idx, axis=-1
            )
            > 1
        )
        min_dr_et2 = ak.argmin(dr_et2.mask[is_duplicate], axis=-1, keepdims=True)
        is_min = ak.fill_none(
            ak.flatten(min_dr_et2[:, None] == fsr_photons.helper_idx, axis=-1), False
        )
        new_lepton_idx = ak.where(
            ~is_duplicate | (is_duplicate & is_min), fsr_photons.lepton_idx, -1
        )
        fsr_photons["lepton_idx"] = new_lepton_idx

        # add fsr_idx field to leptons
        helper_leptons = assign_lepton_fsr_idx(fsr_photons, helper_leptons)

        # for each FSR photon that was selected, we exclude that photon from the isolation sum of all the leptons in the event
        # This concerns the photons that are in the isolation cone and outside the isolation veto of said leptons dR < 0.4 AND dR > 0.01
        selected_fsr_photons = fsr_photons[fsr_photons.lepton_idx > -1]
        muons_matchedfsr_cartesian = ak.cartesian(
            {"muon": muons, "fsr": selected_fsr_photons},
            nested=True,
            axis=1,
        )
        muons_matchedfsr_cartesian_dr = muons_matchedfsr_cartesian.muon.delta_r(
            muons_matchedfsr_cartesian.fsr
        )
        muons_matchedfsr_cartesian_dr_valid = (muons_matchedfsr_cartesian_dr > 0.01) & (
            muons_matchedfsr_cartesian_dr < 0.4
        )
        valid_matchedfsr_pt = ak.sum(
            muons_matchedfsr_cartesian.fsr.mask[muons_matchedfsr_cartesian_dr_valid].pt,
            axis=-1,
        )
        muon_corrected_iso = muons.pfRelIso03_all - (valid_matchedfsr_pt / muons.pt)
        # update pfRelIso03_all field with corrected isolation
        muons["pfRelIso03_all"] = ak.where(
            muon_corrected_iso > 0, muon_corrected_iso, 0.0
        )
        # update 'is_tight' selection for muons
        muons["is_tight"] = muons.is_tight & (muons.pfRelIso03_all < 0.35)

        electrons["pfRelIso03_all"] = ak.zeros_like(electrons.pt)
        # concatenate muons and electrons with corrected iso
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
        leptons = ak.zip(
            {
                "pt": leptons.pt,
                "eta": leptons.eta,
                "phi": leptons.phi,
                "mass": leptons.mass,
                "charge": leptons.charge,
                "pdgId": leptons.pdgId,
                "pfRelIso03_all": leptons.pfRelIso03_all,
                "idx": helper_leptons.idx,
                "fsr_idx": helper_leptons.fsr_idx,
                "is_loose": leptons.is_loose,
                "is_relaxed": leptons.is_relaxed,
                "is_tight": leptons.is_tight,
                "lostHits": leptons.lostHits,
                "sip3d": leptons.sip3d,
                "mvaHZZIso": leptons.mvaHZZIso,
                "isPFcand": leptons.isPFcand,
                "highPtId": leptons.highPtId,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        leptons["idx"] = ak.local_index(leptons, axis=1)
        # assign -1 to FSR lepton_idx associated with the excluded leptons
        index_still_present = ak.any(
            fsr_photons.idx == leptons.fsr_idx[:, None], axis=-1
        )
        new_lepton_idx = ak.where(index_still_present, fsr_photons.lepton_idx, -1)
        fsr_photons["lepton_idx"] = new_lepton_idx

        # select leptons with and without matched FSR photons
        has_matched_fsr_photons = leptons.fsr_idx > -1
        leptons_with_matched_fsrphotons = ak.pad_none(
            leptons[has_matched_fsr_photons], 1
        )
        leptons_without_matched_fsrphotons = leptons[~has_matched_fsr_photons]
        # select FSR photons with matched leptons
        has_matched_leptons = fsr_photons.lepton_idx > -1
        fsr_with_matched_leptons = ak.pad_none(fsr_photons[has_matched_leptons], 1)

        # add p4 field to leptons adding the matched FSR photons
        leptons_with_matched_fsrphotons["p4"] = ak.zip(
            {
                "pt": (leptons_with_matched_fsrphotons + fsr_with_matched_leptons).pt,
                "eta": (leptons_with_matched_fsrphotons + fsr_with_matched_leptons).eta,
                "phi": (leptons_with_matched_fsrphotons + fsr_with_matched_leptons).phi,
                "mass": (
                    leptons_with_matched_fsrphotons + fsr_with_matched_leptons
                ).mass,
                "charge": (
                    leptons_with_matched_fsrphotons + fsr_with_matched_leptons
                ).charge,
                "lostHits": leptons_with_matched_fsrphotons.lostHits,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        leptons_without_matched_fsrphotons["p4"] = ak.zip(
            {
                "pt": leptons_without_matched_fsrphotons.pt,
                "eta": leptons_without_matched_fsrphotons.eta,
                "phi": leptons_without_matched_fsrphotons.phi,
                "mass": leptons_without_matched_fsrphotons.mass,
                "charge": leptons_without_matched_fsrphotons.charge,
                "lostHits": leptons_without_matched_fsrphotons.lostHits,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        # concatenate leptons with and without matched FSR photons after FSR recovery
        leptons = ak.where(
            # concatenate only events with matched FSR photons
            # to avoid including None values from 'leptons_with_matched_fsrphotons'
            ak.sum(leptons.fsr_idx > -1, axis=1) == 0,
            leptons_without_matched_fsrphotons,
            ak.concatenate(
                [leptons_with_matched_fsrphotons, leptons_without_matched_fsrphotons],
                axis=1,
            ),
        )
        self.objects[obj_name] = leptons

    def select_zcandidates(self, obj_name):
        """selects Z candidates for SR and all CRS"""
        # get Z candidates
        zcand = ak.combinations(self.objects["leptons"], 2, fields=["l1", "l2"])
        # check that Z candidates leptons pass the loose id
        zcand = zcand[zcand.l1.is_loose & zcand.l2.is_loose]
        # add Z candidate, p4, pT and idx fields
        zcand["p4"] = zcand.l1.p4 + zcand.l2.p4
        zcand["pt"] = zcand.p4.pt
        zcand["idx"] = ak.local_index(zcand, axis=1)
        # add the Z candidates to objects
        self.objects[obj_name] = zcand

    def select_best_zcandidate(self, obj_name):
        """selects best Z candidate as the one closest to the nominal Z mass"""
        zmass = 91.1876
        best_zcand_idx = ak.argmin(
            np.abs(self.objects["zcandidates"].p4.mass - zmass), axis=1
        )
        best_zcand = self.objects["zcandidates"][
            best_zcand_idx == self.objects["zcandidates"].idx
        ]
        self.objects[obj_name] = best_zcand

    def select_other_relaxed_leptons(self, obj_name):
        """
        selects additional relaxed leptons in the Z+L CR. Adds the mask 'pass_selection' to additional relaxed leptons that pass the analysis selection
        """
        # select best Z candidates
        best_zcands = self.objects["best_zcandidates"]
        # select relaxed leptons (whose idx are different to best Z candidate lepton's idx)
        relaxed_leptons = self.objects["leptons"][self.objects["leptons"].is_relaxed]
        relaxed_leptons_bestzl1_idx = relaxed_leptons.idx != best_zcands.l1.idx[:, None]
        relaxed_leptons_bestzl2_idx = relaxed_leptons.idx != best_zcands.l2.idx[:, None]
        relaxed_leptons_bestz_idx_mask = ak.flatten(
            relaxed_leptons_bestzl1_idx & relaxed_leptons_bestzl2_idx, axis=-1
        )
        relaxed_leptons = relaxed_leptons[relaxed_leptons_bestz_idx_mask]

        # add the 'pass_selection' attribute to 'relaxed_leptons'. It flags relaxed leptons passing the analysis selection
        is_tight = relaxed_leptons.is_tight == ak.ones_like(
            best_zcands.l1.idx[:, None], dtype=bool
        )
        is_tight = ak.flatten(is_tight, axis=-1)
        # ghost removal: ∆R(η, φ) > 0.02 between each of the leptons (to protect against split tracks)
        relaxed_leptons_bestzl1_dr = relaxed_leptons.metric_table(best_zcands.l1)
        relaxed_leptons_bestzl2_dr = relaxed_leptons.metric_table(best_zcands.l2)
        relaxed_leptons_bestz_dr_mask = ak.flatten(
            (relaxed_leptons_bestzl1_dr > 0.02) & (relaxed_leptons_bestzl2_dr > 0.02),
            axis=-1,
        )
        # QCD suppression cut: invariant mass of relaxed lepton and the opposite sign tight lepton from the best Z candidate should satisfy m2l > 4 GeV
        relaxed_leptons_bestzl1_opposite_charge = ak.flatten(
            relaxed_leptons.charge != best_zcands.l1.charge[:, None], axis=-1
        )
        relaxed_leptons_bestzl2_opposite_charge = ak.flatten(
            relaxed_leptons.charge != best_zcands.l2.charge[:, None], axis=-1
        )
        relaxed_leptons_bestzl1_cartesian = ak.cartesian(
            {"lepton": relaxed_leptons.p4, "zl1": best_zcands.l1.p4},
            nested=True,
            axis=1,
        )
        relaxed_leptons_bestzl1_mass = ak.flatten(
            (
                relaxed_leptons_bestzl1_cartesian.lepton
                + relaxed_leptons_bestzl1_cartesian.zl1
            ).mass,
            axis=-1,
        )
        relaxed_leptons_bestzl1_mass_mask = relaxed_leptons_bestzl1_mass > 4
        relaxed_leptons_bestzl2_cartesian = ak.cartesian(
            {"lepton": relaxed_leptons.p4, "zl2": best_zcands.l2.p4},
            nested=True,
            axis=1,
        )
        relaxed_leptons_bestzl2_mass = ak.flatten(
            (
                relaxed_leptons_bestzl2_cartesian.lepton
                + relaxed_leptons_bestzl2_cartesian.zl2
            ).mass,
            axis=-1,
        )
        relaxed_leptons_bestzl2_mass_mask = relaxed_leptons_bestzl2_mass > 4
        qcd_suppression_mask = (
            relaxed_leptons_bestzl1_opposite_charge & relaxed_leptons_bestzl1_mass_mask
        ) | (
            relaxed_leptons_bestzl2_opposite_charge & relaxed_leptons_bestzl2_mass_mask
        )

        # get full pass selection mask
        pass_selection = is_tight & relaxed_leptons_bestz_dr_mask & qcd_suppression_mask
        relaxed_leptons["pass_selection"] = pass_selection

        # add relaxed leptons to objects
        self.objects[obj_name] = relaxed_leptons

    def select_trilepton(self, obj_name):
        self.objects[obj_name] = (
            self.objects["best_zcandidates"].l1
            + self.objects["best_zcandidates"].l2
            + ak.firsts(self.objects["other_relaxed_leptons"])
        )

    def select_zzcandidates(self, obj_name):
        """selects ZZ candidates for SR and CRs"""
        self.objects[obj_name] = make_cand(
            self.objects["zcandidates"], kind="zz", sort_by_mass=True
        )

    def select_zllcandidates_os(self, obj_name):
        """selects Zll candidates for CRs"""
        self.objects[obj_name] = make_cand(
            self.objects["zcandidates"], kind="zll", sort_by_mass=False, os_method=True
        )

    def select_zllcandidates_ss(self, obj_name):
        """selects Zll candidates for CRs"""
        self.objects[obj_name] = make_cand(
            self.objects["zcandidates"], kind="zll", sort_by_mass=False, os_method=False
        )

    def select_best_zzcandidate(self, obj_name):
        """selects best ZZ candidates for SR"""
        self.objects[obj_name] = select_best_zzcandidate(self.objects["zzcandidates"])

    def select_best_1fcr_zllcandidate(self, obj_name):
        """selects best Zll candidates for 3P1F CR"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates"], "is_1fcr"
        )

    def select_best_2fcr_zllcandidate(self, obj_name):
        """selects best Zll candidates for 2P2F CR"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates"], "is_2fcr"
        )

    def select_best_sscr_zllcandidate(self, obj_name):
        """selects best Zll candidates for SS CR"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates"], "is_sscr"
        )

    # 2026-09-21: OS/SS-split-aware variants for hplusc_mva_4class_CR_merged.yaml
    # (pkatris's design -- separate zllcandidates_os (mass>70) / zllcandidates_ss
    # (mass>100) objects, instead of the single "zllcandidates" object the three
    # functions above are hardcoded to). Added as new, separate functions rather
    # than parameterizing the existing ones, so hplusc_mva_4class_CR.yaml's
    # current production usage of the functions above is completely untouched.
    def select_best_1fcr_zllcandidate_os(self, obj_name):
        """selects best Zll candidates for 3P1F CR (OS/SS-split zllcandidates_os)"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates_os"], "is_1fcr"
        )

    def select_best_2fcr_zllcandidate_os(self, obj_name):
        """selects best Zll candidates for 2P2F CR (OS/SS-split zllcandidates_os)"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates_os"], "is_2fcr"
        )

    def select_best_sscr_zllcandidate_ss(self, obj_name):
        """selects best Zll candidates for SS CR (OS/SS-split zllcandidates_ss)"""
        self.objects[obj_name] = select_best_zzcandidate(
            self.objects["zllcandidates_ss"], "is_sscr"
        )

    def select_mass_4e_1fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_1fcr"], "4e"
        )

    def select_mass_2e2mu_1fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_1fcr"], "2e2mu"
        )

    def select_mass_2mu2e_1fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_1fcr"], "2mu2e"
        )

    def select_mass_4mu_1fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_1fcr"], "4mu"
        )

    def select_mass_4e_2fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_2fcr"], "4e"
        )

    def select_mass_2e2mu_2fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_2fcr"], "2e2mu"
        )

    def select_mass_2mu2e_2fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_2fcr"], "2mu2e"
        )

    def select_mass_4mu_2fcr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_2fcr"], "4mu"
        )

    def select_mass_4e_sscr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_sscr"], "4e"
        )

    def select_mass_2e2mu_sscr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_sscr"], "2e2mu"
        )

    def select_mass_2mu2e_sscr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_sscr"], "2mu2e"
        )

    def select_mass_4mu_sscr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zllcandidate_sscr"], "4mu"
        )

    def select_mass_4mu_sr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zzcandidate"], "4mu"
        )

    def select_mass_4e_sr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zzcandidate"], "4e"
        )

    def select_mass_2e2mu_sr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zzcandidate"], "2e2mu"
        )

    def select_mass_2mu2e_sr(self, obj_name):
        self.objects[obj_name] = select_candidate_mass(
            self.objects["best_zzcandidate"], "2mu2e"
        )

    def select_dphi_4e(self, obj_name):
        self.objects[obj_name] = select_candidate_cjet_dphi(
            self.objects["best_zzcandidate"], "4e", self.objects["cjets"]
        )

    def select_dphi_4mu(self, obj_name):
        self.objects[obj_name] = select_candidate_cjet_dphi(
            self.objects["best_zzcandidate"], "4mu", self.objects["cjets"]
        )

    def select_dphi_2mu2e(self, obj_name):
        self.objects[obj_name] = select_candidate_cjet_dphi(
            self.objects["best_zzcandidate"], "2mu2e", self.objects["cjets"]
        )

    def select_dphi_2e2mu(self, obj_name):
        self.objects[obj_name] = select_candidate_cjet_dphi(
            self.objects["best_zzcandidate"], "2e2mu", self.objects["cjets"]
        )

    def select_dphi_inclusive(self, obj_name):
        self.objects[obj_name] = select_candidate_cjet_dphi(
            self.objects["best_zzcandidate"], "inclusive", self.objects["cjets"]
        )

    # --------------------------------------------------------------------------------
    # HWW
    # --------------------------------------------------------------------------------
    def select_hww_leptons(self, obj_name):
        # set 'leptons' by concatenating electrons and muons
        leptons = ak.concatenate(
            [self.objects["muons"], self.objects["electrons"]], axis=1
        )
        leptons = leptons[ak.argsort(leptons.pt, axis=1)]
        self.objects[obj_name] = ak.zip(
            {
                "pt": leptons.pt,
                "eta": leptons.eta,
                "phi": leptons.phi,
                "mass": leptons.mass,
                "charge": leptons.charge,
                "pdgId": leptons.pdgId,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

    def select_hww_zcandidates(self, obj_name):
        self.objects["zcandidates"] = ak.combinations(
            self.objects["leptons"], 2, fields=["l1", "l2"]
        )
        self.objects["zcandidates"].pt = (
            self.objects["zcandidates"].l1.pt + self.objects["zcandidates"].l2.pt
        )

    def select_hww_mll(self, obj_name):
        self.objects["mll"] = transverse_mass(
            self.objects["zcandidates"].l1 + self.objects["zcandidates"].l2,
            self.objects["met"],
        )

    def select_hww_ml1(self, obj_name):
        self.objects["ml1"] = transverse_mass(
            self.objects["zcandidates"].l1, self.objects["met"]
        )

    def select_hww_ml2(self, obj_name):
        self.objects["ml2"] = transverse_mass(
            self.objects["zcandidates"].l2, self.objects["met"]
        )

    def select_candidate_cjet(self, obj_name):
        self.objects["candidate_cjet"] = self.objects["cjets"][
            ak.argmax(self.objects["cjets"].btagDeepFlavCvL, axis=1)
            == ak.local_index(self.objects["cjets"], axis=1)
        ]
