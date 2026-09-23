# H→ZZ + charm (H+c) — analysis pipeline

End-to-end guide for the **`hplusc_mva_4class`** workflow: NanoAOD → parquet →
MVA → datacard → limit. This documents the H→ZZ-specific layer built on top of
the base `higgscharm` framework (filesets, workflow YAML structure, condor
submission) — same framework family as `Chirayu18/higgscharm` (`hww-analysis`
branch), which runs the parallel H+c→WW analysis and is the template this doc
follows.

**Repo (EOS)**: `/eos/user/s/snandaku/Higgscharmnew/higgscharm`
**Repo (AFS)**: `/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm`
**Combine builder**: `/eos/home-s/snandaku/b-hive_ttcc/combine/create_datacards.py`
**Env**: LCG_105 (python/data-loading), separate CMSSW shell for combine itself

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
```

---

## Contents

- [Pipeline overview](#pipeline-overview)
- [0. Prerequisites](#0-prerequisites)
- [1. Process NanoAOD → parquet](#1-process-nanoaod--parquet)
- [2. Postprocess: merge + MVA inference (consolidated)](#2-postprocess-merge--mva-inference-consolidated)
- [3. MVA: training](#3-mva-training)
- [4. MVA: inference](#4-mva-inference)
- [5. Build combine inputs](#5-build-combine-inputs)
- [6. Run the limit](#6-run-the-limit)
- [7. Impact ranking](#7-impact-ranking)
- [Systematics](#systematics)
- [Known traps](#known-traps)
- [Open items / next techniques to port from hww-analysis](#open-items--next-techniques-to-port-from-hww-analysis)

---

## Pipeline overview

**Production** (`hplusc_mva_4class`, historical split path — §4):
```
NanoAOD (DAS/EOS)
   |  runner.py --workflow hplusc_mva_4class          condor, 4 eras
   v
parquet   outputs/hplusc_mva_4class/<era>/<dataset>_<jobid>/base/    per-dataset, per-partition
   |  analysis/postprocess/run_mva_postprocess.py       (NOT run_postprocess.py --postprocess)
   v
mva parquet   outputs/hplusc_mva_4class_scored_v4/<era>/<dataset>_<jobid>/   mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}
   |  b-hive_ttcc/combine/create_datacards.py
   v
datacard  outputs/combine_run3_<window>_v5/{datacard,histograms}.*    1 channel x 4 processes (+ZX)
   |  combine (separate CMSSW shell)
   v
limit
```

**New workflow variants going forward** (consolidated path — §2):
```
NanoAOD (DAS/EOS)
   |  runner.py --workflow <variant>                  condor, 4 eras
   v
parquet   outputs/<variant>/<era>/<dataset>_<jobid>/base/    per-dataset, per-partition
   |  run_postprocess.py --postprocess --mva-inference   (merge -> parquets_<sample>/ -> score, one step)
   v
mva parquet   outputs/<variant>_mvascores/<era>/<sample>/    mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}
   |  b-hive_ttcc/combine/create_datacards_<variant>.py   (copy create_datacards.py, point at new dirs)
   v
datacard -> combine -> limit
```

One structural difference remains from the `hww-analysis` pipeline this
mirrors either way: one channel (`mva_score_Signal` as a single discriminant)
instead of argmax-defined SR+CR channels — kept deliberately, see the note in
[Open items](#open-items--next-techniques-to-port-from-hww-analysis).

---

## 0. Prerequisites

**Grid proxy** — re-init before any condor campaign:
```bash
voms-proxy-init --voms cms --valid 192:00
```

**EOS space** — check before any large campaign:
```bash
eos quota /eos/user/s/snandaku
```
This account runs close to its 2 TB group quota routinely (hit 87.82% once
and caused a real condor failure wave — see the Z+c CR incident in
`Tasks/HcZZ-fake-rate.md`, 2026-08-12). **Never delete anything on EOS
without confirming first.**

**Combine environment** — must be a **separate shell** from the LCG_105 one
above; mixing them corrupts the python interpreter path and crashes combine.
**Now lives inside `Higgscharmnew` itself** (added 2026-08-14, "everything in
one place"), not the older `Analysis/combine/` copy:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /eos/user/s/snandaku/Higgscharmnew/combine/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
```
Same release + combine tag as the pre-existing `/eos/user/s/snandaku/Analysis/
combine/CMSSW_14_1_0_pre4/` copy (`CMSSW_14_1_0_pre4`, `el9_amd64_gcc12`,
`HiggsAnalysis/CombinedLimit` @ `v10.5.0`/commit `af57d49e`) — built fresh
here rather than copied (CMSSW areas don't reliably relocate). **Fully
validated 2026-08-14**, not just a version-string check: ran
`combine -M AsymptoticLimits` on this new environment against the real
`combine_run3_100150_noZX/datacard_100150_fullsyst_noZX.txt` datacard and
got median r = **312.5000**, an exact bit-for-bit match to the documented
reference number.

**Gone as of 2026-09-09**: `/eos/user/s/snandaku/Higgscharmnew/combine/`
no longer exists on disk (checked directly — the `Higgscharmnew/` directory
now only contains `higgscharm/`, not a sibling `combine/`). Cause not dug
into (possibly an EOS quota cleanup — see the standing quota warning above).
**Use the older `/eos/user/s/snandaku/Analysis/combine/CMSSW_14_1_0_pre4/src`
copy instead**, confirmed still present and working 2026-09-09 (reproduced
r=412.0000 bit-exact against the documented JES/JER reference). If the
in-repo copy is ever wanted back, rebuild it fresh rather than looking for
a relocated one — same reasoning as the original 2026-08-14 build.

**Long jobs** run in `tmux`, not `nohup`.

---

## 1. Process NanoAOD → parquet

```bash
python3 runner.py --workflow hplusc_mva_4class --year 2022postEE \
        --submit --eos --output_format parquet
```

Emits **weight-based** shape systematics baked in as extra `weight_<name>Up/Down`
columns (pileup, PS ISR/FSR, LHE scale, LHE PDF/alphaS, c-tag SF) — unlike
`hww-analysis`, there are currently **no object-shift (JES/JER/lepton
scale+res) shifted parquet directories** produced by this workflow. That means
the datacard has no `CMS_scale_j`/`CMS_res_j`/etc. rows at all yet (see
[Open items](#open-items--next-techniques-to-port-from-hww-analysis)).

Also writes a `sumw/<partition>.json` sidecar per MC chunk — the
**true pre-selection** `Σ genWeight`, written unconditionally even for
zero-event chunks (see the sumw fix in
[Known traps](#known-traps)).

Years: `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`.

**Monitor / resubmit:**
```bash
watch condor_q
python3 jobs_status.py --workflow hplusc_mva_4class --year 2022postEE --eos
```

---

## 2. Postprocess: merge + MVA inference (consolidated)

**Updated 2026-08-14 (decision): new workflow variants should use the
consolidated `run_postprocess.py --postprocess --mva-inference` path below,
not the split/skip approach production history left behind.** Kept both
described here since `hplusc_mva_4class_scored_v4` (the directory every
existing r/κc reference number in
`Notes/HcZZ-combine-results-reproducibility.md` was built from) was produced
the old way — don't reinterpret those numbers as having gone through this
path, and don't re-run production through this path without a deliberate
decision to rebuild the reference numbers too.

**Historical state, still true for `hplusc_mva_4class`**: `run_postprocess.py`
supports the same `--postprocess`/`--skipmerging` merge step `hww-analysis`
uses to combine per-dataset parquets into per-process files before MVA
inference, plus a `--mva-inference` flag that runs inference on that merged
output directly (`analysis/postprocess/mva_inference.py::MVAPostProcessor`,
the *same* class `run_mva_postprocess.py` uses — one inference implementation,
two CLI entry points, not a duplicate). **Production never used either flag
together**: `run_mva_postprocess.py` was pointed directly at the raw,
unmerged, per-dataset-per-job-partition parquet tree from step 1 and globs
every `*.parquet` file it finds recursively, skipping the merge step entirely.

That split-path choice was **not a correctness bug** — the two things
`hww-analysis`'s README warns merging protects against were already handled
another way (sumw sidecars, filename-based process matching — see below) —
but it also means the "merge → infer" flow had **never actually been run**
for this analysis before 2026-08-14.

**Now fully working, confirmed by a real run AND a real accuracy check, not just code
inspection**:
```bash
python3 run_postprocess.py -w hplusc_mva_4class -y 2023postBPix \
    --postprocess --output_format parquet --nocutflow \
    --mva-inference \
    --mva-config /eos/user/s/snandaku/Higgscharmnew/higgscharm/models/hc_zzto4l_mw_training_4class_nomass.yml \
    --mva-model  /eos/user/s/snandaku/Higgscharmnew/higgscharm/models/best_model.pt \
    --mva-mass-window
```
**Use `models/best_model.pt` + `models/hc_zzto4l_mw_training_4class_nomass.yml`
(bundled together, documented in `models/README.md`)** — do not substitute a
`b-hive_ttcc/output/TrainingTask/.../best_model.pt` path even if documented
elsewhere as "production" (e.g. `b-hive_ttcc/combine/README.md` documents
`train_loose_oldcwithmassbal` for a *different* consumer, Z+X inference).
Exits 0, scores every dataset in the era, writes a per-sample
`<sample>_mva_scores.coffea` histogram file for each. **`--nocutflow` is
required** for this workflow (see Known traps) — without it the run gets all
the way through scoring-prep and then crashes in the cutflow-summary print,
one step before `--mva-inference` even starts.

Getting here took fixing **ten** independent plumbing/config bugs found by
actually running it against the full production output tree for the first
time, PLUS a real config-composition bug found afterward when the resulting
scores looked wrong (see [Known traps](#known-traps) and second-brain
memories `hww_analysis_pipeline_review` and
`hczz_mva_postprocessor_root_cause` for the complete blow-by-blow).

**Validated with a full-population accuracy check, not a small sample**: all
~40 real datasets in 2023postBPix, 526,011 events, argmax accuracy = **71.04%**
— exceeds the model's own native b-hive `InferenceTask` ground-truth benchmark
(69.64% on its own held-out test set) for the identical checkpoint. Mean
P(Signal|true class) — qqZZ 0.012, ggZZ 0.006, Signal 0.617, Other_Higgs
0.265 — closely matches that same native ground truth (0.068/0.086/0.598/0.231).
One known, real (not a bug) weak point: qqZZ recall is only 28.2%, mostly
misclassified as ggZZ — a genuine model characteristic (these two continuum
processes are kinematically similar, documented previously in this project's
"MVA vs Tagger" work), not a pipeline defect; Signal recall (78.9%, the class
that matters for the physics) is strong.

**Production's existing results are unaffected.** Checked directly:
production's actual `hplusc_mva_4class_scored_v4` parquets (source of every
r=327.0-style reference number) have `mva_score_Signal` on a completely
different scale (~[-0.86, 0.41], Signal mean 0.406, sensible separation from
qqZZ's -0.858) than this tool's `[0,1]`-ish softmax-style score — meaning
production was **never** produced via `MVAPostProcessor`/this consolidated
path at all. Nothing fixed or changed today touched production's scoring.
Reads merged `parquets_<sample>/` dirs (produced by the `--postprocess` merge
step), writes scored output to `outputs/<workflow>_mvascores/<year>/<sample>/`
by default (override with `--mva-output`) — a **different directory
convention** than `hplusc_mva_4class_scored_v4`, so `create_datacards.py
--scored-dir`/`--sumw-dir` need pointing at the new location, not assumed to
match.

**Why this still matters even though sumw/per-process aggregation are already
handled**: `hww-analysis`'s README documents a real incident — "inference
must cover all shift directories, not just nominal... cost 500 [limit]
units" — caught only because counting merged-per-process directories is
trivial (`find ... -name mva -type d | wc -l`, one number to check).
Verifying completeness over the old *per-partition* tree (dozens of numbered
subfolders per dataset, e.g. `HPlusCharm_2022postEE_1` through `_32`) is much
easier to get silently wrong — this consolidation removes that risk for any
new workflow that adopts it, even though it hasn't bitten `hplusc_mva_4class`
in practice (no object-shift directories exist yet to miss, see §1).

---

## 3. MVA: training

**Model**: `MLP_HcZZ_MW_Deep_4class`, 4 classes `[qqZZ, ggZZ, Signal, Other_Higgs]`
**Config**: `hc_zzto4l_mw_training_4class_nomass`
**Training run**: `train_loose_oldcwithmassbal`

Trained with **b-hive** (`b-hive_ttcc/`), separate from the coffea repo — same
tool `hww-analysis` uses for its own training. See `b-hive_ttcc/README.md` for
the training invocation; not duplicated here since it hasn't changed as part
of this doc pass.

---

## 4. MVA: inference (legacy path — still what production uses)

This is what actually produced `hplusc_mva_4class_scored_v4` and every
existing reference number. Kept documented as-is; §2 above is the path for
new workflow variants going forward, not a replacement for this section's
history.

```bash
python3 analysis/postprocess/run_mva_postprocess.py \
    --input-dir  /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class/2022postEE \
    --output-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_scored_v4/2022postEE \
    --config /path/to/b-hive_ttcc/config/hc_zzto4l_mw_training_4class_nomass.yml \
    --model  /path/to/b-hive_ttcc/output/TrainingTask/.../best_model.pt \
    --mass-window-min 90 --mass-window-max 160 \
    --all-eras --workers 8
```

Writes `mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}` into every parquet file it
finds under `--input-dir`, recursively (`Path(input_dir).glob('**/*.parquet')`)
— there is no per-shift-directory iteration to miss the way `hww-analysis`'s
`run_inference.py` has, **because this pipeline doesn't produce shift
directories yet** (§1). If/when object-shift systematics are added, this
script (or its call site) will need the same "verify the count" discipline
`hww-analysis`'s README enforces.

---

## 5. Build combine inputs

```bash
python3 create_datacards.py \
  --scored-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_scored_v4 \
  --sumw-dir   /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class \
  --zx-dir     /eos/user/s/snandaku/Analysis/zx_background_mva_os_90160_featmajor \
  --output     /eos/user/s/snandaku/Analysis/combine_run3_90160_v5
```

One global channel (`hczz`), discriminant = `mva_score_Signal`, 20 quantile
bins by default. Full reproduction commands (all windows/bin-counts tried)
live in `second-brain/Notes/HcZZ-combine-results-reproducibility.md` — that
note is the results/reproducibility index; this README is the pipeline
structure doc, kept deliberately non-duplicative of it.

> **Production switched to the 2D c-tag SF, 2026-08-14** — decision made
> after confirming applying both 1D and 2D SFs together would double-count
> the same correction (a 2D per-category SF already covers a jet's full
> tagger response, it isn't a "companion" to a matching 1D WP). Both
> `hplusc_mva_4class.yaml` (`ctagging_2d: true`, replacing `ctagging: [wp:
> loose]`) and `create_datacards.py` (`CMS_ctag2d` replacing
> `CMS_ctag_b`/`CMS_ctag_c`, `CMS_ctag_light` exclusion retired — the 2D
> scheme has no `eff/(1-eff)` formula to trigger that bug) now reflect this.
> The `atleast_one_cjet` **selection** cut is unchanged (still 1D `loose`
> WP) — matches hww-analysis's own pattern (WP selection + `ctagging_2d`
> only, no 1D SF). **The command above and every existing reference number
> (r=327.0/304.0, κc≈53-57) still reflect the OLD 1D-SF scoring** —
> `hplusc_mva_4class_scored_v4` was produced before this switch. A real
> reprocessing campaign (condor, currently blocked — see Known traps) is
> needed before a new, valid combine number exists under the 2D scheme.

---

## 6. Run the limit

```bash
# (combine env from §0, separate shell)
cd /eos/user/s/snandaku/Analysis/combine_run3_90160_v5
combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001 --rRelAcc 0.00001 \
  datacard_mva_with_zx.txt -n HcZZ_v5_check
```

`--run blind` suppresses the fake Asimov-derived "Observed" line; tight
`rAbsAcc`/`rRelAcc` kept as a standing habit (confirmed bit-identical vs.
default tolerance in this project, but the SMEFT sister-project's own note
flagged default tolerance as sometimes too loose to resolve small
differences between similar-sensitivity models).

Current production reference (no-ZX): median r = 327.0 [90,160] full-syst /
316.5 stat-only. Full table (all windows, bin counts, κc conversions):
`second-brain/Notes/HcZZ-combine-results-reproducibility.md`.

---

## 7. Impact ranking

`b-hive_ttcc/combine/impact_ranking.py` — added 2026-08-14, ported/adapted
from `hww-analysis`'s `scripts/combine/make_impact_plot.py`. Same method
(cheap approximation, not a real CombineHarvester `-M Impacts` profile fit):
freeze one nuisance at a time, rerun `AsymptoticLimits`, rank by
`|median_r_frozen - median_r_nominal|`. Adapted to auto-parse nuisance names
directly from a plain-text datacard (`<name> <shape|lnN> ...` rows) instead
of `hww-analysis`'s yaml-driven `combine:` block, since `create_datacards.py`
here writes flat datacards, not a `WorkflowConfigBuilder`-mediated one.

```bash
# (combine env from §0)
cd /eos/user/s/snandaku/Higgscharmnew/combine/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
python3 /eos/home-s/snandaku/b-hive_ttcc/combine/impact_ranking.py \
  --card datacard_mva_with_zx.txt \
  --dir  /eos/user/s/snandaku/Analysis/combine_run3_90160_v5 \
  --plot
```

**Fully validated end-to-end 2026-08-14**: `parse_nuisances()` correctly extracted
all 13 real nuisances from `combine_run3_100150_noZX/datacard_100150_fullsyst_noZX.txt`,
and a real 2-nuisance smoke run (`--nuisance CMS_pileup CMS_ctag_b`) against
that same real datacard, using the new in-repo combine environment (§0),
correctly reproduced the known nominal median r = 312.5000 and produced
sane per-nuisance deltas (`CMS_pileup`: −1.0/−0.32%, `CMS_ctag_b`: +0.0/0%).
`--plot` needs matplotlib+uproot (not in the combine CMSSW python by
default — run that step from an LCG env against the written
`impact_ranking.json` if needed) — not yet exercised. Output:
`<dir>/impact_ranking.json` (+ `.png` with `--plot`).

A nuisance whose limit **improves** when frozen usually signals degeneracy
with a free-floating rateParam (here: `ZX_rate`), not a broken nuisance —
same caveat `hww-analysis`'s README notes for their `rate_tt`.

**Update 2026-08-30 — `--real` flag added, the cheap proxy now warns on every
run.** Cross-checking against Felix Heyen's thesis methodology (Sec 4.5.2)
found this script's default freeze-and-compare method actually **disagrees**
with the real profile-likelihood method on this analysis, not just
approximates it: on the ctag2d `[100,150]` no-ZX `_v5` card (r=299.5), the
proxy ranked `CMS_ctag2d`/`CMS_eff_e_reco_below20` a near-tie for #1
(−3.51%/−4.01%) and read `CMS_eff_m_id` as exactly 0.00 impact; the real
method ranks `CMS_ctag2d` clearly #1 (impact 143.6, ~3.2× the #2 value) and
finds `CMS_eff_m_id` a small but nonzero 0.029. **The ranking order and a
supposedly-exactly-zero nuisance both changed** — this proxy is fine for
quick iteration but should not be the number quoted in a paper, AN, or
final slide.

Added `--real`: runs the actual `combineTool.py -M Impacts` 3-stage
profile-likelihood workflow (`text2workspace.py` → `--doInitialFit
--robustFit 1` → `--doFits --robustFit 1 --parallel N` → collect
`impacts_real.json` → `plotImpacts.py`, with a best-effort `pdftoppm` PNG
conversion), matching Felix's convention of injecting the Asimov dataset
with signal at the nominal expected-limit strength (not background-only,
which is what this project's one prior "real" Impacts run, 2026-08-16,
used instead). Needs `combineTool.py`/`plotImpacts.py` on `PATH` (same
CombineHarvester install as plain `combine`) and a wide
`--setParameterRanges` (`--r-range`, default `0,1000` — the CombineHarvester
default `r∈[0,20]` is far too narrow for this analysis's real r~O(100–600)
sensitivity and causes boundary-pinned, meaningless per-nuisance fits, the
same mistake caught and fixed 2026-08-16 for the manual 3-stage workflow).

```bash
# (combine env from §0)
python3 /eos/home-s/snandaku/b-hive_ttcc/combine/impact_ranking.py \
  --card datacard_no_zx.txt \
  --dir  /eos/home-s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_100150_ctag2d_v5 \
  --real [--expect-signal 299.5] [--r-range 0,1000] [--parallel 4]
```

Without `--expect-signal`, the script runs a nominal `AsymptoticLimits`
first to find the injection point automatically. **Verified end-to-end
2026-08-30**: rerunning against the `_v5` card reproduced the manually-run
comparison numbers exactly (`CMS_ctag2d` impact 143.6333, `CMS_eff_m_id`
0.0294, best-fit `r̂ = 299.500 (+180.369/-194.149, 68%)`), and the default
(no `--real`) path was re-verified afterward with `--skip-existing` to
confirm it still produces its usual (proxy) output unchanged. Full
writeup: `second-brain/Tasks/HcZZ-fake-rate.md`, "Update 2026-08-30".

---

## Systematics

Declared in `create_datacards.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts
(weight-based, read as `weight_<name>Up/Down` — same mechanism
`hww-analysis` calls `combine.shape_systematics`).

| kind | examples | notes |
|---|---|---|
| weight shapes | `lhe_pdf`, `lhe_alphaS` (excluded, sign bug fixed at source 2026-08-21 but non-retrofittable onto existing parquets — see update below), `scalevar_muR`, `scalevar_muF`, `ps_isr`, `ps_fsr`, `CMS_pileup`, `CMS_ctag2d`, `CMS_eff_e_reco_{below20,20to75,above75}` | per-process usable subset, not uniform — see `USABLE_SYST` and Known traps |
| **muon efficiency SF (ID only)** | **wired 2026-08-21** (`id: loose`), real single-file smoke test passed end-to-end — see the dated update below. `iso`/`trigger` deliberately left off (see rationale in the workflow yaml). Doesn't change any existing r number — needs a full reprocessing to take effect (condor-blocked). |
| **electron ID SF** | not applied (`id: false`) | documented, deliberate — smaller likely magnitude than the muon gap |
| object shifts | `CMS_scale_j`/`CMS_res_j`/`CMS_scale_m`/`CMS_res_m` **live in the datacard as of 2026-09-08** (`hplusc_mva_4class_ctag2d_jecshifts` workflow, `--jecshifts-dir` in `create_datacards_ctag2d_100150.py`; independently reprocessed-and-rescored object-shift samples, not weight-ratio shapes). `CMS_scale_j` ranks #6 of 22 nuisances by impact — see the "CURRENT" callout above and memory `hczz_jesjer_datacard_complete` | `hww-analysis` has live `CMS_scale_j/e/m`, `CMS_res_j/e/m` from JES/JER/lepton-scale shift directories — this row is now at parity for jet/muon shifts (lepton = electron scale/res still not ported) |
| lnN | `lumi_Run3` (1.4%), `pdf_gg`, `QCDscale_ggZZ`/`QCDscale_qqZZ`, `BR_HZZ4l` (2%), `ZX_norm` (30%) | values cross-validated against HIG-24-013 2026-08-21 (lumi 1.4% exact match); `pdf_qq` deliberately NOT added — qqZZ's real `lhe_pdf` shape already covers it, a separate lnN would double-count |
| rateParam | `ZX_rate` | Z+X floats freely from data, same pattern as `hww-analysis`'s `tt` |
| MC stat | `autoMCStats <threshold>`, **off by default since 2026-08-19** — confirmed non-convergent at threshold 10/50/100 (11.5h, zero quantiles) | `--automcstats-threshold` CLI flag opts back in for testing; real open gap, not a not-yet-tried item |

`CMS_ctag_light` is **excluded entirely**, not just per-process-gated — see
Known traps.

### `ctag2d` variant (active as of 2026-08-17)

> **Default mass window changed to `[100,150]` GeV, 2026-08-18** (tighter S/√B,
> 0.0054 vs 0.0043 at `[90,160]`). `combine/run_pipeline.sh` with no args now
> builds `[100,150]` by default (`--window 90160` for the older window), and
> the production default at this window remains **MC-only** — Z+X is not
> wired into `run_pipeline.sh`'s own `--with-zx` preset, only built via direct
> script invocation. **A true 4-era with-ZX number at `[100,150]` now exists
> against the current `_v8` + JES/JER production (2026-09-11): r=480.0
> median/κc=85.89** — see the dated update below for the full chain. `[90,160]`'s
> own with-ZX number (r=329.0, from the OS/featmajor production) remains
> available via `--window 90160 --with-zx` but is now stale relative to the
> `[100,150]` result above. **`lhe_alphaS` removed 2026-08-19
> pending a bug fix (see the dated update below).**
> **CURRENT (as of 2026-09-08, no-ZX): r=412.0 median / kappa_c=74.11, 22
> systematics (18 shape/lnN + JES/JER), `_scored_v8` + `_jecshifts_scored_v2`**
> — `--jecshifts-dir` wired into `create_datacards_ctag2d_100150.py`
> (`CMS_scale_j`/`CMS_res_j`/`CMS_scale_m`/`CMS_res_m`, independently
> reprocessed-and-rescored object-shift samples, not weight-ratio shapes);
> `CMS_scale_j` ranks #6 of 22 nuisances (impact 13.64), individually
> significant even though the 4 added systematics only move the overall
> limit ~0.2% (r=411.0→412.0). See second-brain memory
> `hczz_jesjer_datacard_complete` for the full writeup. Without JES/JER:
> r=411.0/kappa_c=73.93 (2026-09-08, chunk-dedup fix — a pipeline bug was
> dropping 13–27% of real event chunks; see memory
> `hczz_chunk_dedup_bug_and_v8_correction`). A fresh plain/kappa/3-ratio
> discriminant comparison against this production (2026-09-09,
> no-JES/JER scope) found all three within ~0.6% of each other — see
> second-brain `Notes/ctag2d-3ratio-kappa-comparison-note.md` /
> memory `hczz_ctag2d_3ratio_kappa_comparison`.
> Everything below this point in the chain is historical/superseded, kept
> for the investigation trail: r=413.0/kappa_c=74.28 (2026-09-02, `_scored_v7`,
> the `CMS_ctag2d` L0 up/down swap fix — see memory
> `hczz_ctag2d_l0_updown_swap_fix`, well-supported but not POG-confirmed);
> r=299.0/kappa_c=54.52 (2026-08-31, 17 systematics, `_scored_v6`) was itself
> superseded by the swap fix — the preceding 3-jet-cap fix (also 2026-08-31)
> was only a partial mitigation, confirmed retrospectively; r=219.0/40.64
> (2026-08-29) was **RETRACTED** — a duplicate-job-output bug, not a real
> improvement; r=299.5/kappa_c=54.60 (2026-08-30) was superseded by the
> 3-jet-cap fix (negligible 0.17% shift at the time). Also superseded:
> r=301.0/kappa_c=54.86, 16 systematics (2026-08-27, muon SF added but
> pre-`lhescale.py`-fix); r=309.5/15 systematics (2026-08-19, pre-muon-SF);
> r=311.5/16 systematics (2026-08-17, buggy `lhe_alphaS` still included).

Scripts: `combine/scripts/create_datacards_ctag2d.py` ([90,160] GeV, canonical),
`create_datacards_ctag2d_100150.py` ([100,150] GeV copy, `MASS_WINDOW` hardcoded
at line 117 — no CLI flag for the window), `create_datacards_ctag2d_3ratio.py`
(joint nested-tree 3-ratio discriminant variant of the same pipeline, see below).
All three share the same `SYSTEMATICS`/`USABLE_SYST` dicts and sumw/xs-scaling
logic as `create_datacards_ctag2d.py` — edit all three together when changing
the systematics model.

**19 systematics total** (was 18 as of 2026-09-02; `lhe_alphaS` re-included
2026-09-11 for qqZZ/Other_Higgs once its abs()-before-symmetrizing bug was
confirmed fixed — see the "Update 2026-09-11: PDF/scale systematics audit"
section below): 6 `lnN` + 13 shape.

| kind | name | size/processes | notes |
|---|---|---|---|
| lnN | `lumi_Run3` | 1.4%, all | |
| lnN | `QCDscale_gg` | 3.9%, ggZZ+Signal | **added 2026-08-28**, matches HIG-24-013 Table 15's gg-initiated QCD-scale value exactly; replaces the old 5% `pdf_gg` placeholder's mislabeled half |
| lnN | `pdf_gg` | 3.2%, ggZZ+Signal | **value corrected 2026-08-28** to HIG-24-013's real PDF(gg) number (was a 5% round placeholder) |
| lnN | `kfactor_ggZZ` | 10%, ggZZ | **renamed 2026-08-28** from `QCDscale_ggZZ` — value (10%) was already numerically correct, just named after the wrong physics quantity (it's HIG-24-013's gg→ZZ k-factor, not a QCD-scale term) |
| lnN | `BR_HZZ4l` | 2%, Signal+Other_Higgs | **added 2026-08-17**, sourced from Felix Heyen thesis Appendix D; no per-event weight column exists, pure rate lnN |
| lnN | `QCDscale_qqZZ` | 4%, qqZZ | **added 2026-08-17**, same source; separate from `kfactor_ggZZ`, not a duplicate. **Decoupled 2026-09-11** from qqZZ's `scalevar_muR`/`scalevar_muF` shape templates (previously double-counted the same ~4% normalization via both an lnN and an un-decoupled shape template — see the dated update below) |
| shape | `lhe_pdf`/`scalevar_muR`/`scalevar_muF` | qqZZ only | excluded for Signal/ggZZ/most of Other_Higgs — see per-process reasons below. `scalevar_muR`/`muF` are normalization-decoupled for qqZZ as of 2026-09-11 (rescaled to nominal integral, acceptance+shape only — see `QCDscale_qqZZ` row above) |
| shape | `lhe_alphaS` | qqZZ, Other_Higgs | **re-included 2026-09-11** — the abs()-before-symmetrizing bug (removed everywhere 2026-08-19) was fixed in `lhepdf.py` 2026-08-28 but the datacard exclusion was never revisited until now. Excluded for ggZZ/Signal (separately degenerate, unrelated to this bug — see dated update below) |
| shape | `ps_isr`/`ps_fsr`/`CMS_pileup`/`CMS_ctag2d` | all 4 processes | |
| shape | `CMS_eff_e_reco_20to75`/`above75`/`below20` | all 4 processes | **added 2026-08-17** — already computed and stored in the scored parquets (`weight_CMS_eff_e_reco_*_<year>Up/Down`), never wired into `USABLE_SYST` before. Verified sane (≤6%, symmetric) for every process incl. Signal/HPlusBottom via direct per-event ratio check — a detector-level correction, unaffected by the private-LHE-sample bugs below. |
| shape | `higgs_plus_c` | **Other_Higgs only** | **added 2026-09-02**, alongside the `CMS_ctag2d` L0 swap fix. Higgs+HF (ggH/VBF heavy-flavor-composition) uncertainty from `analysis/corrections/higgs_hf.py` (ported from HiggsDNA, AN-23-102 §7.1); present in scored parquets since `higgsHFWeight: true` (2026-08-28) but never wired into `USABLE_SYST` until now. Deliberately narrow-scoped — flat ±50% on ggH/VBF events with a true gen-level c-jet, exactly 1.0 elsewhere, column absent entirely from qqZZ — so excluded for qqZZ/ggZZ/Signal explicitly and only left in for the pooled `Other_Higgs` process (which includes ggH/VBF). Real impact ranking: #11, 7.51. |
| shape | `CMS_eff_m_id` | all 4 processes | **added 2026-08-27** — muon ID efficiency SF (loose WP), wired into the workflow yaml 2026-08-21 but only reached a real scored-parquet tree and a combine run on 2026-08-27, once the ctag2d reprocessing campaign reached usable completeness. Detector-level correction like `CMS_eff_e_reco` above, not an LHE weight — usable for every process. |

**Why `lhe_pdf`/`lhe_alphaS`/`scalevar_muR`/`scalevar_muF` are missing for some
processes** (full reasoning already inline as comments above `USABLE_SYST` in
the script; summarized here for discoverability):
- **Signal**: `lhe_pdf` — root cause found (missing `1/sqrt(N-1)` NNPDF
  normalization, fixed in source for future reprocessing) but a downstream
  patch on existing parquets still leaves ~88% PDF uncertainty (vs qqZZ's
  4.3%) — excluded until real reprocessing lands. `lhe_alphaS` — not a bug,
  Signal's PDF replica set genuinely has only 101 members, no dedicated
  αs-variation member exists. `scalevar_muR`/`muF` — separate, still-undiagnosed
  bug: Up and Down both shift the *same* direction (should be symmetric ±),
  specific to this private sample's `LHEScaleWeight` branch.
- **ggZZ**: simplest case — no `LHEScaleWeight` branch exists at all in these
  MC samples; `lhe_pdf`/`lhe_alphaS` degenerate (nothing to vary).
- **Other_Higgs**: a *pooled* process — official centrally-produced samples
  (ggH/VBF/WH/ZH/ttH/bbH) are fine (properly antisymmetric), but the private
  `HPlusBottom` component (same private LHE chain as Signal) shows the
  identical two bugs — `lhe_pdf` blown up to +857%/−657%, `scalevar_muR`/`muF`
  same-direction Up/Down. HPlusBottom is only ~4% of the yield but dominates
  the shape variation — this is what caused an earlier Combine "Bogus norm"
  crash once a masking clip was removed. **`lhe_alphaS`'s "not pathological,
  no same-direction Up/Down" note here was itself wrong — corrected
  2026-08-19, see the dated update below: same-direction Up/Down for
  `lhe_alphaS` is universal (a code bug affecting every process with
  LHEPdfWeight members), not a property that happened to distinguish
  HPlusBottom from the official samples.**

**Not added (checked, deliberately skipped as redundant)**: `weight_scalevar_muR_muF`
and `weight_lhe_pdf_alphaS` (combined variants) exist in the parquets too, but
adding them alongside the already-used separate `muR`/`muF`/`pdf`/`alphaS` rows
would double-count the same uncertainty.

**Muon ID SF: resolved 2026-08-27** (was "not addable without upstream work" as
of this writing) — `weight_CMS_eff_m_id_<year>Up/Down` now exists in a real
scored production tree (`hplusc_mva_4class_ctag2d_scored_v2`) and is wired
into the datacard as `CMS_eff_m_id`, see the table above and the 2026-08-27
dated update below. Muon isolation/reco SF and electron ID/isolation SF are
still not applied (deliberate scope choices, see the 2026-08-21 dated update
for the muon isolation rationale). JES/JER object-shift systematics still
need new shift-directory production (see Open items #4).

### Update 2026-08-19: `jec_shifts` bug fixed, muon scale/resolution shift production added, `higgsHFWeight` smoke-tested, `autoMCStats` threshold made configurable

Picked back up after a laptop crash killed the prior shell mid-session — **nothing
was lost**: all edits from that session (`higgsHFWeight` wiring, `higgs_hf.py`,
`muon_ss.py`) were already on disk in the EOS checkout, just uncommitted. Confirmed
directly (`git status`) before doing anything else.

**Bug found and fixed: `object_corrector_manager`'s `"jec_shifts"` branch used to
`return apply_jerc_shifts(...)` immediately**, silently skipping `muon_ss`/
`electon_ss` entirely for `hplusc_mva_4class_ctag2d_jecshifts.yaml` even though the
workflow yaml lists both under `corrections.objects` — every event in that
workflow's output was missing the muon scale/smearing correction and (for
2022–2023) the electron scale/smearing correction, silently. Caught while wiring
in muon shift production (needed a place to hook `apply_muon_ss_shifts` into).
**Fixed**: `muon_ss`/`electon_ss` nominal corrections now run *before* JES/JER
shift production, not skipped. Order doesn't matter physically —
`apply_jerc_shifts` reads `events[met_field_key]` fresh at call time (confirmed by
reading `get_corrected_jets_with_shifts`), and independent MET deltas (Type-1 JEC,
muon SS) commute under addition — so every JES/JER variant automatically inherits
the nominal muon/electron correction too, with no extra plumbing.
Every workflow run through this path **before** 2026-08-19 is missing muon SS
(and, for 2022/2023, electron SS) — re-run before trusting any yield/shape from
`hplusc_mva_4class_ctag2d_jecshifts.yaml` dated earlier.

**Muon scale/resolution shift production added** (`CMS_scale_m_<year>`,
`CMS_res_m_<year>`, MC only) — `analysis/corrections/muon_ss.py::
apply_muon_ss_shifts()`, mirroring `jerc.py::apply_jerc_shifts`'s
`(collections, shift_name)` list contract. Reuses the file's existing
`pt_scale_var`/`pt_resol_var` functions (already implemented, previously unused)
for the Up/Down math. MET propagation reuses `met.py::update_met`'s own tested
px/py-delta formula via a throwaway array rather than re-deriving that algebra —
deliberate, to avoid adding a second, independently-drifting copy of a formula
after this project's history of exactly that kind of bug (ratio-clip/sign-bug
precedents elsewhere in `analysis/corrections/`).

`object_corrector_manager` now assembles JES/JER and muon-scale/res variants into
one combined list, **explicitly carrying `Jet`/`MET`/`Muon` in every entry even
where unchanged from nominal** — required because `base.py`'s shift loop mutates
`events` cumulatively across iterations rather than resetting to nominal each
time; an entry that omitted a key would silently inherit whatever the previous
loop iteration left behind. `base.py` extended to apply a `"Muon"` override the
same way it already applies `"Jet"`/`"MET"`.

**Smoke-tested for real** (`apptainer exec` into the production coffea image,
one real `GluGluHtoZZto4L`/2023postBPix file, `hplusc_mva_4class_ctag2d_jecshifts`
workflow — needed `--bind /eos --bind /afs --bind /run/user/<uid> --bind /cvmfs`
plus `X509_USER_PROXY` pointed at a real voms proxy; none of those binds are
automatic in this environment, unlike a real lxplus node). Ran clean, exit 0, all
9 shift directories written (`base` nominal + `CMS_scale_j/res_j_2023{Up,Down}` +
`CMS_scale_m/res_m_2023{Up,Down}`). Verified from the actual parquet output, not
just "no crash":
- JES/JER variants: event count changes sensibly (183–201 vs nominal 193 — jet-
  dependent selection responding to shifted jet kinematics), `jet_pt` shifts,
  `lepton_pt` stays close to nominal.
- Muon variants: `jet_pt` is **bit-identical** to nominal across all 4 muon
  variants (44.575, confirms `Jet` correctly stays pinned to nominal in every
  muon-shift entry — the "no stale state across loop iterations" design works).
  `lepton_pt` shows the expected small (sub-percent) shift. Event count unchanged
  (193 in all 4 — muon momentum shifts here are small enough to essentially never
  cross a selection threshold, unlike JES/JER).
- `weight_higgs_plus_c` / `weight_nominal` ratio is exactly **1.5 (Up) / 0.5
  (Down)** for the 13/1728 events carrying a genuine c-flavour GEN jet, exactly
  1.0 for every other event and for the nominal weight itself — matches the
  documented ±50% design exactly, correctly scoped.

**Not yet done**: `weight_higgs_plus_c` is computed in the parquet but **not yet
wired into `create_datacards_ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts** —
next concrete step once a real (non-smoke-test) `jecshifts` production run exists.
JES/JER and muon-scale/res shift *production* is code-complete and validated;
building a datacard that actually USES these as shape systematics (reading from
the new per-shift subdirectories) is separate, not-yet-started work.

**AFS synced 2026-08-19**: needed the exact copy in AFS as well since jobs get
submitted from there (EOS runs postprocess/inference/combine, AFS is what
condor submission actually uses), so copied the
full correction chain + `base.py` + both `ctag2d` workflow yamls from EOS to
AFS: `correction_manager.py`, `ctag.py`, `ctag2d.py`, `higgs_hf.py`, `jerc.py`,
`jec_params_correctionlib.yaml`, `lhepdf.py`, `muon_ss.py`, `utils.py`,
`processors/base.py`, `hplusc_mva_4class_ctag2d.yaml`,
`hplusc_mva_4class_ctag2d_jecshifts.yaml`. Verified `py_compile`-clean on AFS
after copying.

**`autoMCStats` threshold made a CLI flag** (`--automcstats-threshold`, default
10, unchanged behavior) in both `create_datacards_ctag2d.py` and `_100150.py` —
to test whether raising it past the current sparse-bin degeneracy (several bins
have MC-stat errors 10–90× the bin content, noted 2026-08-18) stabilizes the fit.
Rationale: `autoMCStats <threshold>` compares each bin's *effective* MC event
count (`n_eff = (Σw)²/Σw²`, not raw entry count) against the threshold — above it,
Combine uses the cheap merged Barlow-Beeston-**lite** Gaussian nuisance per bin;
below it, the exact per-process Poisson/gamma treatment. Default 10 is standard
combine practice, but this analysis's Signal/HPlusBottom carry ~25% negative-
weight MC events (documented private-LHE-generator issue), which makes `n_eff` a
noisier estimate of real statistical power than in a normal positively-weighted
sample — bins that nominally clear `n_eff≥10` may still sit on very few raw
events, right where the lite Gaussian approximation is weakest. Raising the
cutoff routes more of those marginal bins to the exact treatment instead.
**Result not yet in hand** — a threshold=10 baseline reproduction (same
`--merge-bin-ranges "2-4,7-10" --skip-zx` production recipe) was still running
past 19+ minutes at last check (started 2026-08-19 ~16:38, PID 3003391 in that
shell — PID won't survive a machine reboot, but the log path below will), well
beyond this pipeline's normal <2-minute `AsymptoticLimits` runtime — itself
informative (the sparse-bin degeneracy is a real fit-performance cost, not just
a theoretical concern) but not yet a comparable number for the 50/100 tests.
Session ended before it finished (by choice, not a crash) — **resume here**:

```bash
# 1. Check whether the threshold=10 baseline finished:
tail -60 /tmp/snandaku/automcstats_thr10.log
# (built into $HOME/.../combine/outputs/automcstats_test_thr10/ -- if it's not
#  there or the log looks incomplete, it may have been killed by a shell/session
#  restart; just relaunch it, same command as below with 10.)

# 2. Once 10 has a result (or after relaunching it), run 50 and 100 the same way,
#    each into its own --output dir so they don't collide:
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm/combine
./run_pipeline.sh --merge-bin-ranges "2-4,7-10" --skip-zx \
  --automcstats-threshold 50 --output "$PWD/outputs/automcstats_test_thr50" \
  > /tmp/snandaku/automcstats_thr50.log 2>&1 &

./run_pipeline.sh --merge-bin-ranges "2-4,7-10" --skip-zx \
  --automcstats-threshold 100 --output "$PWD/outputs/automcstats_test_thr100" \
  > /tmp/snandaku/automcstats_thr100.log 2>&1 &

# 3. Compare median r (and total runtime / whether it converged cleanly at all)
#    across 10/50/100 -- runtime and convergence behavior matter here as much
#    as the r value itself, since the whole point is testing fit STABILITY,
#    not just getting a number.
```

**Result, 2026-08-19: definitive non-convergence at all three thresholds, sweep
abandoned.** Relaunched all three (10/50/100) fresh — the prior session's `/tmp`
logs didn't survive the laptop crash. Let them run **11.5 hours** (Aug 18 23:04 →
Aug 19 10:34), confirmed genuinely CPU-bound the whole time (`ps` CPU-time tracked
wall-clock 1:1 at ~99.6% CPU, not sleeping/deadlocked) via two checks roughly 25
minutes apart plus the final one. **None of the three produced even a first
`Expected N%: r < ...` quantile line** — not "slow", genuinely non-converging,
and critically, **raising the threshold did not help**: threshold=100 (which
should route almost every bin to the exact per-process treatment instead of the
fragile Barlow-Beeston-lite Gaussian) was exactly as stuck as threshold=10. This
rules out the sweep's own hypothesis (marginal bins near a low threshold being
the culprit) — the degeneracy isn't about *which* bins get the lite treatment,
it runs deeper, plausibly in how `autoMCStats`'s Gaussian nuisances interact with
this analysis's ~25% negative-weight Signal/HPlusBottom MC regardless of the
threshold. Killed all three (`kill -TERM`, confirmed clean) rather than let them
burn CPU indefinitely. **Conclusion: raising the threshold alone does not fix
it — that specific idea is closed, not the underlying problem.** `autoMCStats`
stays out of the production datacard for now, but **this is a real, unresolved
gap, not an accepted permanent state**: per-bin MC statistical uncertainty on
the templates currently has no nuisance covering it at all. A CMS analysis note
needs some accounting for this; "omit it because it doesn't converge" is a
documented workaround, not a resolution. **Candidate next steps, not yet
started, roughly in order of expected effort-to-payoff**:
1. Identify exactly which bins have the 10–90× error/content ratio and merge
   just those further (the pipeline already merges some ranges for the
   negative-bin issue) — if `n_eff` clears the threshold everywhere, plain
   `autoMCStats` may converge normally without needing any threshold games.
2. Attack the root cause directly: Signal/HPlusBottom's ~25% negative-weight
   MC fraction (same likely root cause as the already-diagnosed `lhe_pdf`/
   `scalevar_muR/muF` blowups) makes `n_eff` an unreliable proxy for real
   statistical power — fixing how negative weights are summed/handled could
   resolve several problems at once, not just this one.
3. Manual per-bin `shapeN`/gamma nuisances for just the worst offending
   bins/processes instead of the automatic whole-channel treatment.
4. Debug the fit itself before assuming it's structural — untried:
   `--cminDefaultMinimizerStrategy 0`, `--X-rtd MINIMIZER_analytic`, looser
   `--rAbsAcc`/`--rRelAcc`, or plain `MultiDimFit` instead of the full
   `AsymptoticLimits` CLs scan, to isolate whether the CLs scan itself is the
   bottleneck versus the underlying likelihood being genuinely broken. The
   11.5-hour test only ruled out "threshold alone fixes it."

### Update 2026-08-19: `lhe_alphaS` sign bug found — same-direction Up/Down was
never real physics, removed from every datacard pending a fix

While investigating why `lhe_alphaS` ranked an unexpectedly high #3 in the
16-systematic Impacts (impact_r=17.6 — see the rebuild in the previous
section), checked the actual templates directly rather than trusting the fit:
both `qqZZ` (±1.5%) and `Other_Higgs` (±2.7%) move in the **same direction**
together — Up always raises both, Down always lowers both, smoothly and
without spikes across all 20 bins. That's the tell: a real, independent physics
effect on two different production processes shouldn't be forced into lockstep
like that.

**Root cause, found in `analysis/corrections/lhepdf.py:60`**:
```python
delta_alpha = 0.5 * np.abs(w_as_high - w_as_low)
w_up_alpha = 1 + delta_alpha
w_down_alpha = 1 - delta_alpha
```
The `np.abs()` destroys the *sign* of the difference between the two αs
members (`w_as_high` = LHEPdfWeight member 102, αs=0.120; `w_as_low` = member
101, αs=0.116) before symmetrizing. Since `delta_alpha` is forced non-negative,
`w_up_alpha >= 1` and `w_down_alpha <= 1` for **every event of every process,
unconditionally** — "Up" is mathematically guaranteed to raise the yield and
"Down" to lower it, regardless of which direction the real αs variation
pushes that event's weight. This is why `qqZZ` and `Other_Higgs` were seen
moving together: not a shared physics effect, an artifact of the formula. Any
other process with a 103-member `LHEPdfWeight` (Signal/ggZZ don't have one —
separately degenerate, unaffected) would show the identical manufactured
pattern.

Contrast with `lhe_pdf` immediately above it in the same function (lines
43-55): that one is *supposed* to be a symmetric envelope —
`sqrt(Σ(w_k-w0)²)` is the correct PDF4LHC treatment for ~100 Hessian-eigenvector
replica members, where "symmetric uncertainty, direction not physically
meaningful" is the right convention. αs has only **2** members, not 100 — the
physically correct treatment is a genuinely *signed* shift (higher αs → member
102's actual weight, lower αs → member 101's), not an absolute-value envelope.
The two-members case wrongly reused the many-eigenvectors convention.

**This also retroactively corrects a wrong conclusion recorded earlier in this
file** (the `USABLE_SYST` per-process reasoning above): `lhe_alphaS` was
described as "kept for Other_Higgs: larger for HPlusBottom than the official
samples, but not pathological (no sign flip, no same-direction Up/Down)" — that
comparison was never meaningful, since same-direction Up/Down turns out to be
universal, not something distinguishing HPlusBottom from the official samples.

**Not yet fixed, deliberately** — excluded from `USABLE_SYST` instead
(`create_datacards_ctag2d.py`, `create_datacards_ctag2d_100150.py`,
`create_datacards_ctag2d_3ratio.py`, all three kept in sync) pending
verification that a signed fix (drop the `np.abs()`) genuinely changes
direction for some events rather than just relabeling the same magnitude —
that check wasn't done before documenting, to avoid publishing an unverified
"fixed" number.

**Updated production limit, 15 systematics, `lhe_alphaS` excluded**: median
r=309.5 full-syst, no-ZX, `[100,150]` (was r=311.5 with the buggy systematic
included — a small, expected shift since removing one shape systematic
slightly changes the fit, not a sign the old number was wildly wrong, just
built on an incorrect nuisance). Converged cleanly in <2 minutes (confirms this
whole investigation was never about `autoMCStats`/convergence — that's a
separate, still-open issue, see above).

**Impact ranking rebuilt, 15 systematics** (same real `combineTool.py -M
Impacts` profile-likelihood method as before, not the cheap approximation):

| Rank | Nuisance | impact on r |
|---|---|---|
| 1 | `CMS_ctag2d` | 71.2 |
| 2 | `CMS_eff_e_reco_below20` | 28.2 |
| 3 | `BR_HZZ4l` | 12.0 |
| 4 | `lumi_Run3` | 9.6 |
| 5 | `CMS_eff_e_reco_20to75` | 7.3 |
| 6 | `CMS_pileup` | 6.5 |
| 7 | `ps_fsr` | 5.4 |
| 8 | `QCDscale_qqZZ` | 4.0 |
| 9 | `ps_isr` | 3.0 |
| 10 | `scalevar_muF` | 2.7 |
| 11 | `scalevar_muR` | 1.3 |
| 12 | `QCDscale_ggZZ` | 0.7 |
| 13 | `lhe_pdf` | 0.4 |
| 14 | `pdf_gg` | 0.3 |
| 15 | `CMS_eff_e_reco_above75` | 0.2 |

`CMS_ctag2d` remains the dominant nuisance by a wide margin — unaffected by
this fix, reinforcing that resolving its Down-variation asymmetry (kept as a
documented caveat, see the Known traps entry) is the highest-value remaining
systematics work. Output:
`combine_run3_100150_ctag2d_pipeline_run/impacts_100150_15syst_noalphaS.{json,pdf}`.

**Also caught and fixed while rebuilding, unrelated regression**: the
`autoMCStats` code from the section above turned out to default to **on**
(`hczz autoMCStats 10` silently written into every datacard unless a value is
explicitly passed) — directly contradicting the standing decision to keep it
out, and confirmed by watching a freshly-rebuilt production datacard hang
again during this same session. Fixed: `--automcstats-threshold` now defaults
to `None` (no line written at all) in both `create_datacards_ctag2d.py` and
`create_datacards_ctag2d_100150.py`; pass an explicit value to opt back in for
testing.

**Next step, not started**: verify the signed fix (`delta_alpha = 0.5 *
(w_as_high - w_as_low)`, dropping `np.abs()`) actually flips direction for a
real subset of events before re-including `lhe_alphaS` and rebuilding again.

---

### Update 2026-08-21: `lhe_alphaS` signed fix applied (non-retrofittable),
muon efficiency SF wired, one real regression caught and fixed by an actual
smoke test

**Full reproducibility reference** (exact commands, exact diffs, exact expected output for
everything below): `second-brain/Notes/HcZZ-systematics-audit-2026-08-21.md`. The reusable
smoke-test script itself is `smoketest_muon_sf.py` at this repo's root (EOS+AFS, identical) —
run it directly to re-verify the muon-SF wiring end to end.

**`lhe_alphaS` signed fix applied**: dropped the `np.abs()` per the "next
step" above. Checked whether this could be retrofitted onto already-scored
parquets (real `2023postBPix` ctag2d-scored file schemas) — only the final
post-abs() `weight_lhe_alphaSUp/Down` columns survive, no raw
`LHEPdfWeight[:,101]`/`[:,102]` components to recompute the true sign from.
**Confirmed non-retrofittable**, same situation as the already-documented
`lhe_pdf` normalization bug. `lhe_alphaS` stays excluded from `USABLE_SYST`
until a real reprocessing exists; rebuilt the production `[100,150]` no-ZX
card with the fix present (still excluded) — r=309.5000, bit-identical to
the pre-fix baseline, as expected (the fix only benefits a future
reprocessing).

**Real regression caught by an actual smoke test, not a config check**:
applying this fix initially left `w_as_low`/`w_as_high` undefined —
`NameError: name 'w_as_high' is not defined` — a genuine bug that would have
crashed every future reprocessing (this function runs for every 103-member
`LHEPdfWeight` event, i.e. essentially all MC). Never caught by the earlier
audit pass because that pass only rebuilt the datacard from already-scored
*parquets*, which never call this NanoAOD-level correction function at all —
only a real single-file `run_uproot_job` smoke test (see below) exercises
this code path. Fixed by restoring the two extraction lines
(`w_as_low = pdfweights[:, 101]`, `w_as_high = pdfweights[:, 102]`) before
the signed-shift formula, in both EOS and AFS `lhepdf.py`.

**Muon efficiency SF (ID only) wired into both ZZ→4l workflow yamls**
(`hplusc_mva_4class.yaml`, `hplusc_mva_4class_ctag2d.yaml`, EOS+AFS),
closing the gap found in the 2026-08-21 systematics-completeness audit
(second-brain memory `hczz_systematics_completeness_audit`) — no ZZ→4l
workflow previously had a `muon:` key at all.
```yaml
muon:
  - id: loose
  - iso: false
  - trigger: false
```
- **`id: loose`**, not `tight`: this analysis's actual muon selection
  (`zzto4l.yaml` → `is_loose` = isGlobal|isTracker + kinematic/IP cuts,
  `is_tight` = `is_loose` + `is_relaxed` (SIP3D<4) + `isPFcand`) is the
  standard CMS H→ZZ→4l lepton recipe, which is much closer to POG **loose**
  muon ID (PF muon + isGlobal|isTracker) than POG **tight** ID (which
  additionally requires normalized χ², station-match, and tracker-layer
  cuts this selection never checks). Using `tight` here would apply an SF
  for a selection efficiency tighter than what's actually cut on —
  overcorrecting, not fixing a gap. `correctionlib` safely no-ops (SF=1)
  for muons below the SF binning's pT floor (10–15 GeV depending on nano
  version; see `MuonWeights.get_id_weights`/`unflat_sf`), so this
  analysis's `pt>5` GeV muons are only corrected above that floor — no
  crash risk, no silent extrapolation.
- **`iso: false`, deliberately not wired**: this analysis's isolation is a
  bespoke FSR-photon-corrected combined relative isolation
  (`select_zzto4l_leptons` in `analysis/selections/object_selections.py`),
  not a standard PF-relIso working point at a matching cone/threshold — the
  generic `NUM_*RelIso_DEN_*ID` correctionlib SFs don't correspond to this
  cut. Applying one anyway would risk a **new** mismatch bias, not fix one;
  left open rather than silently misapplied.
- **`trigger: false`**: matches the electron block's own choice and the
  `hww.yaml` sibling; a real multi-path (SingleMu/DiMu/TriMu/SingleEle/
  DiEle/MuEle) trigger SF needs combinatorial OR-of-paths efficiency
  modeling beyond a single flag flip — out of scope for this pass.

**Validated with a real end-to-end smoke test, not just YAML/config
inspection**: ran the actual `WorkflowConfigBuilder` → confirmed
`corrections_config["event_weights"]["muon"]` parses to
`{"id": "loose", "iso": False, "trigger": False}` exactly as
`correction_manager.py`'s muon branch expects (list-of-single-key-dicts →
merged dict, same mechanism the working `electron:` block already uses).
Then ran a **real single-file `processor.run_uproot_job`** (one
`GluGluHtoZZto4L`/2023postBPix file, `hplusc_mva_4class_ctag2d` workflow,
interactively in LCG_105 — no apptainer needed this time since coffea
imports directly there) end to end: exit clean, 191 events selected,
`weight_CMS_eff_m_id_2023`/`Up`/`Down` written to the output parquet with
sane values (mean SF 0.999, spread ≤0.5% per event, exactly matching this
framework's `weight_<name>Up/Down` convention that `create_datacards_
ctag2d*.py` already knows how to read).

**Two incidental bugs found and fixed while setting up this smoke test**
(both pre-existing, unrelated to the muon-SF work itself, only surfaced
because this was the first real interactive `run_uproot_job` test run
outside apptainer in a while):
1. The `lhe_alphaS` `NameError` above.
2. `analysis/data/nnlo_ps/` was missing an `__init__.py` (unlike its sibling
   `analysis/data/jec/__init__.py`), making it an implicit namespace
   package — Python 3.9's `importlib.resources.open_text` resolved it
   inconsistently in this interactive environment, constructing a wrong
   path (`/eos/home-s/.../NNLOPS_reweight.json`, missing the
   `analysis/data/nnlo_ps/` subdirectory entirely) and crashing
   `add_nnlops_weight`. Added the matching empty `__init__.py` (EOS+AFS) —
   turns it into a regular package, same fix pattern as the existing `jec/`
   sibling. Apparently never bitten real condor production (different
   environment setup there), but a real latent fragility worth having
   fixed regardless.

**At the time of this update, this did not yet change any existing combine
number** — `weight_CMS_eff_m_id_*` only existed in that session's throwaway
smoke-test parquet, not in any production scored-parquet tree. See the
2026-08-27 dated update immediately below for the full reprocessing +
datacard + combine run that finally used it for real.

### Update 2026-08-27: muon SF reaches a real combine number for the first
time — new production reference r=301.0, kappa_c=54.86 (16 systematics)

**Full reproducibility reference**: `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-27 ~19:30 CEST"; memory
`hczz_systematics_completeness_audit` §16-18 for the full chain of findings
that unblocked this (proxy-auth fix, `jobs_status.py` completeness-tool
fix, real-vs-apparent job completeness).

**Context**: the muon SF gap (§2 of the 2026-08-21 audit) was wired into the
workflow yamls the same day, but stayed blocked on reaching a real scored
production tree — the ctag2d reprocessing campaign that would carry
`weight_CMS_eff_m_id_*` into production appeared stuck at 65-90% completeness
for several days (2026-08-25 through 2026-08-27), traced through a site-wide
XRootD outage, then an AFS-proxy auth bug, before a same-session forensic
dig found the deeper cause: `jobs_status.py`'s completeness check couldn't
tell "job ran fine, zero events survived a tight selection" (expected
physics, especially for background MC and even some data streams) from
"job never ran" — real completeness was already 97-100% per era. Fixed
`jobs_status.py` (AFS checkout) to fall back to checking the always-written
flat `{dataset}_{N}.coffea` file before declaring a job missing.

**Rebuilt the MVA-scored parquets from scratch** (`analysis/postprocess/
run_mva_postprocess.py`, LCG_105 env, the documented `models/best_model.pt`
+ `hc_zzto4l_mw_training_4class_nomass.yml` bundled pair) into a new
directory, `hplusc_mva_4class_ctag2d_scored_v2` (old `_scored` dir from
2026-08-16 left untouched) — all 4 eras, 5035 parquet files, **0 errors**,
2,883,041 total events. Confirmed `weight_CMS_eff_m_id_2022/2023Up/Down`
present for the first time ever in a scored production tree.

**Added `CMS_eff_m_id`** (era-dependent, same convention as `CMS_pileup`/
`CMS_ctag2d`) to `SYSTEMATICS`/`USABLE_SYST` in all three ctag2d datacard
scripts — usable for every process (detector-level SF, not an LHE weight,
none of the private-sample exclusions apply). 16 systematics total now.

**Ran the standard production preset** (`run_pipeline.sh --merge-bin-ranges
"2-4,7-10" --skip-zx`, i.e. `[100,150]`, surgical rebin, no-ZX) against the
new scored dir:

**Result: r < 301.0 (median, 50% CL), kappa_c = 54.86.** Full CLs band:

| CL | r | kappa_c |
|---|---|---|
| 2.5% | 154.0273 | 29.35 |
| 16.0% | 209.1421 | 38.92 |
| 50.0% | 301.0000 | 54.86 |
| 84.0% | 441.3765 | 79.20 |
| 97.5% | 624.6505 | 110.96 |

Tightens modestly from the pre-muon-SF reference (r=309.5, 2026-08-19, 15
systematics) — expected direction/magnitude for one real systematic plus a
much more complete underlying production. Negative-bin scan: 15/74
histograms with a single ~1e-6-magnitude negative bin at index 0 (2 new
entries are `CMS_eff_m_idUp/Down`, same tiny benign magnitude as the rest,
same pattern documented in the 2026-08-18 negative-bins update). A ROOT
`TNetXNGFile::Open [3001]` error appeared in the combine log right after
CMSSW startup but didn't block or crash the fit — looks like a benign ROOT
plugin-manager probe, not investigated further.

**Re-ran impact ranking** (§7 method, freeze-and-compare) against this new
datacard: `CMS_eff_m_id` has **zero measurable impact** on the median
(`delta_r = 0.00`, exactly) — dominated by `CMS_eff_e_reco_below20`
(-3.82%) and `CMS_ctag2d` (-3.49%), same top-2 as the pre-muon-SF ranking.
Consistent with its small per-event magnitude (mean SF 0.999, <=0.5% spread,
per the 2026-08-21 smoke test) — correctly wired and included, just
genuinely small at this method's resolution, not a sign anything is wrong.
Output: `combine/outputs/combine_run3_100150_ctag2d_muonSF/impact_ranking.
{json,png}`.

**Then resubmitted the remaining genuinely-missing jobs** (9 ctag2d across
3 eras + 6 `ztoee` + ~103 `zplusl_ss`, using the fixed `jobs_status.py` to
correctly identify only the real gaps) — draining as of this update; a
still-more-complete number is expected once that lands and the pipeline is
rerun again.

**Nothing old was overwritten** — new scored dir
(`hplusc_mva_4class_ctag2d_scored_v2`) and new output dir
(`combine/outputs/combine_run3_100150_ctag2d_muonSF/`), both additive.

---

### Update 2026-08-28: `lhescale.py` Up/Down swap found and fixed; Signal/
HPlusBottom `lhe_pdf`/`scalevar_muR/muF` root cause confirmed from raw
NanoAOD (not a code bug); ggZZ exclusion confirmed structural

**Full reproducibility reference**: `second-brain/Notes/HcZZ-systematics-
audit-2026-08-28.md`; narrative in `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-28 (later still): lhe_pdf/scalevar_muR/muF audit".

**Audited every up/down systematic in `analysis/corrections/*.py`** for the
same class of bug as the historical `lhe_pdf`/`lhe_alphaS` issues. Every file
except `lhescale.py` delegates Up/Down to a correctionlib-named systematic
string (`up`/`down`, `systup`/`systdown`, `sfup`/`sfdown`, `up_Total`/
`down_Total`) — no custom-index risk. `jerc.py`'s JES/JER shift production
(`apply_jerc_shifts`) checked and confirmed correct (`fac=+1/-1` around
`jesunc` for JES; JER delegates to correctionlib's named systematic).
`partonshower.py`'s `PSWeight` indices checked against the standard CMS
NanoAOD producer convention — correct.

**Real bug found in `lhescale.py`**: confirmed `coffea.analysis_tools.
Weights.add()`'s signature is `(name, weight, weightUp, weightDown)`. Per
the file's own docstring index table (index[1]/[7]=muR down/up,
index[3]/[5]=muF down/up, index[0]/[8]=both down/up), the code passed
`weightUp`/`weightDown` in the **opposite** order for all three of
`scalevar_muR`/`scalevar_muF`/`scalevar_muR_muF` — Up and Down were
physically swapped, for every process, since this file was written.
`scalevar_muR_muF` also had a missing `/nom` ratio normalization (the only
one of the three without it). **Fixed at the source**, both EOS+AFS
checkouts, `ast.parse`-checked. Affects qqZZ/Other_Higgs's current
`USABLE_SYST` entries in every existing scored parquet — needs a
reprocessing to take effect, same non-retrofittable class as the
`lhe_alphaS` fix above.

**Signal/HPlusBottom root cause, confirmed from real NanoAOD (not the
2026-08-03 parquet-level retrofit math)**: pulled real events directly via
`uproot` from the regenerated `HPlusCharm_2023postBPix`/
`HPlusBottom_2023postBPix` production (`hplusc_htozz1`/`hplusb_htozz1`
paths). Confirmed index[4] (central ratio point) is exactly `1.000000`,
std=0, for every event — rules out the "branch normalization is broken"
hypothesis open since 2026-08-03. But every *other* grid index sits at
mean/median ≈1.5–2.5× (not the expected ≈0.8–1.2×): Signal
muR-up=1.96×/muR-down=2.24×, HPlusBottom muR-up=1.71×/muR-down=2.03× —
median tracks mean closely, not an outlier artifact. `delta_pdf`
recomputed from the raw 101/103-member `LHEPdfWeight` arrays with the
current fixed formula: Signal mean=1.08/median=1.27, HPlusBottom
mean=0.85/median=1.00 — same ~90-130% range as the earlier ~88% figure,
now independently confirmed from raw LHE weights.

**Explanation**: both datasets are tagged `MuRFScaleDynX0p50` — central
scale = half of some dynamical reference. Every other 9-point-grid entry
multiplies renscfact/facscfact onto that already-halved central scale, so
the "central" weight sits at an outlier point in its own envelope rather
than the middle — a genuine generator-setup artifact, not a downstream
formula bug. Wiring a ~100-200% lnN/shape for Signal would dominate every
other nuisance and misrepresent this as real theory uncertainty; keeping
the exclusion. Real fix (if pursued) is regenerating with a standard
central-scale definition — new item in Open items below, not a datacard
change.

**ggZZ exclusion confirmed structural**: pulled real
`GluGluToContinto2Zto2E2Mu` events — `LHEScaleWeight`/`LHEPdfWeight`
branches exist in the schema but are zero-length for every single event.
Not present-but-degenerate — genuinely empty; this MCFM-based sample was
never produced with these vectors filled.

---

**3-ratio discriminant variant** (`create_datacards_ctag2d_3ratio.py`): joint
$(r_1,r_2,r_3) = (P_s/(P_s{+}P_{qqZZ}), P_s/(P_s{+}P_{ggZZ}), P_s/(P_s{+}P_{OH}))$,
nested/conditional quantile tree binning (ported from `b-hive_ttcc/combine/
create_datacards_3ratio.py`'s "nested" mode), same 16-systematic set. Beats
plain score on full-syst at both mass windows (1–3% tighter) but not stat-only
(flat-to-slightly-worse) — opposite of the pattern documented in
`second-brain/Notes/SM-3ratio-kappa-decomposition-note.md` for the old
ctag_light/with-ZX test (4–10% better, both tiers). **Root cause found**:
rerunning the identical code against the *old* (`hplusc_mva_4class_scored_v4`,
scored 2026-08-04) parquets reproduces that old reference bit-exactly —
proves the 3-ratio construction/binning code itself is correct. The
`ctag2d`-scored parquets (scored 2026-08-16) sit on the other side of a
2026-08-14 MVA-inference bugfix (era-aware jet-tag one-hot boundaries, feature-
composition fix — see the config-composition trap below) that shifted Signal/
Other_Higgs classifier scores ~47–49% (qqZZ/ggZZ unaffected — those bugs are
jet-tagging-specific). Mechanism: each ratio $r=P_s/(P_s+P_\text{bkg})$
saturates in $[0,1]$ as $P_s$ grows, so it structurally captures less of a
large classifier improvement than the unbounded raw score does (raw
Signal-vs-qqZZ separation grew 71% from that fix; each ratio axis only grew
26–30%) — with a fixed 20-bin budget, plain score's full-resolution single
axis exploits that improvement fully, the 3-ratio's $2{\times}2{\times}5$
split across 3 axes cannot. Verified directly from per-event scores, not
hypothesized.

**New issue found while adding the electron-reco systematics above (2026-08-17,
not yet fixed)**: `CMS_ctag2d`'s Down variation is suspiciously flat (≈1.00)
for *every* process, while Up shows a real 9–25% shift — not the expected
symmetric ± pattern for a shape systematic (qqZZ: Up=1.088/Down=1.0001; ggZZ:
1.089/1.002; Signal: 1.249/1.028; Other_Higgs: 1.132/0.999). Looks like
`weight_CMS_ctag2d_*Down` may not be applying a real down-shift. Matters
because `CMS_ctag2d` ranks #2–3 of the nuisances in the impact-ranking plots —
that ranking may be affected. **Root-caused 2026-08-18**: the Up/Down weight-combination
logic itself is standard/correct — the real source is placeholder calibration values
(`central=1.000, up_Total=3.000, down_Total=0.300`) in statistically-empty (flavor, WP)
corners of the official `flavTaggingSF_<year>.json.gz`, amplified by `CTag2DCorrector`
taking `ak.prod` across every selected jet rather than one. **Decision (2026-08-18):
keep the all-jets `ak.prod` scope** — it matches the 1D `CTagCorrector`
this replaces, keeping pre/post-migration yields directly comparable, so the asymmetry
stays a documented caveat rather than triggering a scope change. Restricting to one jet
would likely smooth it out but was explicitly rejected as a bigger, separate
validation task that changes what the corrector represents.

**Revisited 2026-08-31 — the 71.2 impact this decision was made against turned out to be
roughly half the real value (143.6 from the real profile-likelihood fit); scope changed to
a 3-jet cap.** See the dated update below.

---

### Update 2026-08-29: fresh systematics correctness re-audit — no new bugs,
`lhescale.py` fix confirmed correctly implemented in code

**Full reproducibility reference**: `second-brain/Notes/HcZZ-systematics-
audit-2026-08-29.md`; narrative in `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-29".

Re-read every `analysis/corrections/*.py` Up/Down call from the current
checkout (not a recall of the 2026-08-28 audit) and cross-checked against
`combine/scripts/create_datacards_ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST`.
Confirmed `lhescale.py`'s fix is correctly implemented (index[7]/[5]/[8] as
Up, index[1]/[3]/[0] as Down, `/nom` present on all three variables) and
diff-identical between the EOS and AFS checkouts. `lhepdf.py`, `ctag.py`,
`ctag2d.py`, `pileup.py`, `partonshower.py`, `muon.py`, `electron.py` all
re-verified correct. `electron_ss.py`/`muon_ss.py`/`met.py`/`jetvetomaps.py`
confirmed to carry no weight systematics at all (object-level corrections
only, correctly out of scope). `USABLE_SYST` confirmed identical across all
three production datacard scripts (`create_datacards_ctag2d.py`,
`..._100150.py`, `..._3ratio.py`). No new bugs found.

**No code changed this session** — verification only. `lhescale.py`'s fix was
pushed to `Saranya-Nandakumar/higgscharm` (`HcZZ-zx-estimation`, `075478d`)
earlier the same day; this audit confirms that push is correct as-is.

---

### Update 2026-08-29 (later): full reprocessing campaign — `lhescale.py` fix
and gg-lnN split now take effect for the first time; new production reference
**r=219.0 / kappa_c=40.64** (was r=301.0 / kappa_c=54.86)

The 2026-08-28/29 fixes (`lhescale.py` Up/Down swap; `QCDscale_gg`/`pdf_gg`/
`kfactor_ggZZ` lnN split) only existed in source code / datacard scripts until
now — neither had been through a real condor reprocessing + rescoring +
combine cycle. Ran the full chain end to end:

1. **Condor reprocessing** (`hplusc_mva_4class_ctag2d`, MC datasets only, all
   4 eras, via `submit_condor.py` direct per-dataset calls — same pattern as
   the 2026-07-31 sumw-fix resubmission). Two follow-up resubmission rounds
   for congestion-driven XRootD failures (`Socket timeout`/`Operation
   expired` against `cms-xrd-global.cern.ch` — the same well-documented
   flakiness as 2026-08-25 through 08-28, no new root cause). **Final
   completeness: 100% of every `PROCESS_MAPPING`-relevant dataset** (Signal,
   qqZZ, ggZZ, Other_Higgs) across all 4 eras — the only persistent gaps are
   in datasets `get_process_from_path()` maps to `None` (single-top,
   diboson/triboson, DY HT-binned, plus the already-known-broken
   `bbH_Hto2Zto4L`), which don't feed the datacard at all.
2. **MVA rescoring** (`run_mva_postprocess.py`, `--no-mass-window`, all 4
   eras) into a new `hplusc_mva_4class_ctag2d_scored_v4` directory — **9,503
   files, 0 errors, 4,329,108 total events** (supersedes `_v2`/`_v3`, both
   pre-dating the `lhescale.py` fix).
3. **Datacard + combine rebuild** (`run_pipeline.sh`, default `[100,150]`
   preset, `--scored-dir` pointed at `_v4`): **median r = 219.0**, full band
   2.5%/16%/50%/84%/97.5% = 112.9/152.7/219.0/319.4/450.4, **kappa_c = 40.64**
   (band 22.18/29.11/40.64/58.05/80.76). Reproduced bit-identically on a
   second independent combine invocation. A **~27% tightening** vs the
   pre-fix `r=301.0`/`kappa_c=54.86` (2026-08-27) — larger than the
   "shape-only, small lnN-equivalent magnitude" expectation from the
   `lhescale.py` fix alone, so some of the movement is the underlying
   production being fresher/more complete, not purely the Up/Down relabel;
   not decomposed further this session.
4. **Impact ranking rebuilt** (`impact_ranking.py --plot`, all 17
   systematics): top-2 unchanged from the 15/16-syst cards —
   `CMS_eff_e_reco_below20` (−4.57%), `CMS_ctag2d` (−3.88%). `CMS_eff_m_id`
   confirmed **exactly zero impact** (Δr=0.00), consistent with the
   2026-08-28 freeze-and-compare proxy — this is now a real `-M
   AsymptoticLimits`-based ranking of the full systematics list, not a
   proxy. Every nuisance "improves the limit when frozen" — expected for a
   blind/Asimov fit with **no floating rateParam** in this no-ZX card (unlike
   the with-ZX card's `ZX_rate` degeneracy caveat), not evidence of a
   problem.
5. **Two transient-EOS-read incidents, self-resolved, no code bug**: the
   very first `combine` call on the freshly-written `histograms_no_zx.root`
   intermittently threw `RuntimeError: Bogus norm 0.0 for ... ggZZ,
   systematic CMS_eff_m_id Down` (preceded by a ROOT `TBufferFile::
   ReadClassBuffer ... wrong version: 256` read error) on 3 of 4 attempts —
   same class of issue as the 2026-08-28 git-race note. `md5sum` confirmed
   the EOS file itself was never corrupted; copying it to local `/tmp` disk
   made every subsequent `combine`/`impact_ranking.py` call succeed
   deterministically. **Takeaway for next time**: if `combine` throws a
   `Bogus norm`/ROOT-version error immediately after a datacard/histogram
   rebuild, retry once or copy the ROOT file to local disk before assuming a
   real content bug.

Outputs: `combine/outputs/combine_run3_100150_ctag2d_v4/` (datacard, ROOT
histograms, `impact_ranking.json`/`.png`). `--scored-dir` default in all
three datacard scripts (`create_datacards_ctag2d.py`, `_100150.py`,
`_3ratio.py`) updated from `_scored_v2` to `_scored_v4`.

**How to apply**: r=219.0/kappa_c=40.64 was believed at the time to be the
production reference — **RETRACTED 2026-08-30** (duplicate-job-output bug,
see below); **current reference is r=299.0/kappa_c=54.52** (2026-08-31,
`_scored_v6`, see the "Update 2026-08-31 (later)" section near the end of
this file). `second-brain/Notes/HcZZ-combine-results-reproducibility.md`
still needs updating with this chain — not yet done. The `[90,160]`/with-ZX
and `_3ratio` variants have **not** been rerun against any post-`_scored_v2`
tree yet — their existing numbers still reflect `_scored_v2`.

---

### Update 2026-08-30: independent re-verification before slide update —
item 5 above was wrong, the `histograms_no_zx.root` corruption was real and
persisted; r=219.0/kappa_c=40.64 now genuinely confirmed reproducible

> **SUPERSEDED same day — see "Update 2026-08-30 (later)" below.** The
> corruption fix and template audit described in this section are still
> correct, but the r=219.0/kappa_c=40.64 number they reproduced was itself
> wrong for an unrelated reason (a duplicate-parquet-counting bug). Do not
> use r=219.0/kappa_c=40.64. **This section's own r=299.5/kappa_c=54.60
> replacement was in turn superseded 2026-08-31** by the ctag2d 3-jet-cap
> fix's reprocessing (negligible 0.17% shift) — **current reference is
> r=299.0/kappa_c=54.52**, see "Update 2026-08-31 (later)" near the end of
> this file.

Asked to re-verify r=219.0/kappa_c=40.64 (every systematic's Up/Down
templates, the impact plot, the number itself) before it went into the
slides — not a recall of the account above, an actual fresh re-derivation.
**Item 5's "self-resolved, no code bug" framing from 2026-08-29 turned out
to be wrong**: the `histograms_no_zx.root` sitting in
`combine_run3_100150_ctag2d_v4/` today **deterministically crashed combine,
5/5 attempts, including from a fresh local-disk copy** — `RuntimeError:
Bogus norm 0.0 for channel hczz, process ggZZ, systematic CMS_eff_m_id
Down`. A second independent reader (`uproot`, no ROOT/CMSSW) agreed
`h_ggZZ_CMS_eff_m_idDown` was genuinely malformed. That rules out the
transient-EOS-read-race explanation (which predicts intermittent failures
fixed by a local-disk copy, not a 100%-reproducible one) — the 2026-08-29
session's local-disk workaround evidently fixed the run *for that session
only* (hence the clean `combine_result.log`/`impact_ranking_full.log` from
that evening) without the fix ever landing back on the EOS copy, which
stayed broken. Root cause of the underlying corruption still not
determined — treated as an isolated write/read fault on that one file, not
a code bug (no other object in the file was affected).

**Full 75-object Up/Down template audit** (all 4 processes × 17
systematics) found nothing else wrong: no missing pairs, no Up==Down
duplicates, no dead (identical-to-nominal) systematics, no zero-integral
templates. The known Signal/HPlusBottom negative-bin dip (bin 0, ~1e-6 to
2e-5, see the negative-weight-fraction row in the Known traps table) is
present but consistent across nominal and every variation — not new.

**Fix**: reran `run_pipeline.sh --merge-bin-ranges "2-4,7-10" --skip-zx`
against `_scored_v4`, overwriting the corrupt file in place (same
`combine_run3_100150_ctag2d_v4/` directory, same paths this README already
documents). Fresh combine run, repeated twice: **r=219.0/kappa_c=40.64,
full band 112.9/152.7/219.0/319.4/450.4 / 22.18/29.11/40.64/58.05/80.76 —
exact match to the 2026-08-29 numbers, now genuinely reproducible from a
healthy file.** Impact ranking rebuilt fresh: identical top-2
(`CMS_eff_e_reco_below20` −4.57%, `CMS_ctag2d` −3.88%), `CMS_eff_m_id` still
exactly 0.00; new `impact_ranking.png` is md5-identical to the one already
committed in `second-brain/slides/AN_HcZZ_figures/` — that PNG was fine all
along, only the datacard ROOT file was bad. `h_ggZZ_CMS_eff_m_idDown` reread
via `uproot` post-rebuild: healthy, Up/Down (4.1645/4.1570) mirror
symmetrically around nominal (4.1607).

**Revised takeaway for item 5 above**: a `Bogus norm`/ROOT-version error
right after a fresh datacard/histogram rebuild is *not* always the transient
EOS read-race it was in 2026-08-28's case — if a local-disk retry also fails
deterministically and a second independent reader (`uproot`) agrees the
object is malformed, stop retrying and rebuild the datacard from the scored
parquets instead; don't assume the workaround that fixed one session's
`combine` run also fixed the file it read from.

Slides (`second-brain/slides/analysis_overview_hczz.tex`,
`combine_summary_hczz.tex`) already had the r=219.0/kappa_c=40.64 numbers
committed 2026-08-29 (`d5b34e0`) — added "independently re-verified
2026-08-30" addenda documenting this incident rather than changing any
numbers. Full writeup: `second-brain/Tasks/HcZZ-fake-rate.md`, "Update
2026-08-30".

---

### Update 2026-08-30 (later): r=219.0 RETRACTED — duplicate-parquet-
counting bug found and fixed; true production reference **r=299.5 /
kappa_c=54.60**, essentially unchanged from r=301.0/54.86

Reaction to the "confirmed reproducible" verdict above: the stat-only limit
was also 300-something, so this 219 looked suspicious. Correct instinct — a
full-syst limit *below* a stat-only baseline built on the same production is
not physically possible (systematics only ever loosen an expected limit
relative to stat-only). Chased it down rather than trusting the
reproducibility check above, which only proved the number was *stable*, not
that it was *correct*.

**Same-file stat-only check** (`combine ... --freezeParameters
allConstrainedNuisances` on the identical `_v4` `datacard_no_zx.txt`):
stat-only median = 194.5. Full-syst (219.0) vs stat-only (194.5) → ratio
1.126 — directionally fine, no bug in that ordering. **But** 194.5 is
itself ~30% tighter than the *old* card's stat-only (279.0) — far more than
the credited fixes (pure shape/lnN relabels, no yield effect) could
explain. Compared the `rate` lines directly:

| Process | Old (08-28 → r=301.0) | `_v4` (08-30 → r=219.0) | Change |
|---|---|---|---|
| Signal | 0.053225 | 0.082333 | +54.7% |
| ggZZ | 2.944030 | 4.160709 | +41.3% |
| qqZZ | 36.199311 | 58.928004 | +62.8% |
| Other_Higgs | 55.088346 | 66.748218 | +21.2% |

Every process jumped 20–63% — directly contradicting this same session's
own earlier claim that its resubmission round only recovered `None`-mapped
datasets that never feed the datacard.

**Root cause**: the raw postprocess tree
(`outputs/hplusc_mva_4class_ctag2d/<era>/<dataset>_<N>/base/*.parquet`) is
read via `glob('**/*.parquet')` — recursively, unglobbed against prior
partitions — by both `run_mva_postprocess.py` (MVA scoring) and
`create_datacards_ctag2d_100150.py`'s `load_sumw()` (sumw denominator).
Every condor resubmission round (2026-08-14, 08-25, 08-29 — all documented
in earlier sections of this file) re-splits the file list into
**freshly-numbered partition directories** (`HPlusCharm_2022postEE`, `_1`,
`_10`, `_20`, `_30`, ...) instead of replacing older ones, so the same
physical source file (identified by a GUID embedded in its parquet
filename) accumulates across campaigns and gets summed multiple times.
Verified on `2022postEE`: **HPlusCharm 926 files / 468 unique GUIDs
(2.0×)**, **ZZto4L 1169 files / 208 unique GUIDs (5.6×)** — qqZZ's largest
yield jump lines up with its largest duplication factor. This has likely
been inflating every ctag2d-line combine result since at least 2026-08-14,
not just this pair.

**Fix** (dedupe at read time, non-destructive — old files stay on disk,
filtered out at read time): added `dedupe_parquet_files_by_source()`
(GUID-from-filename, keep latest write by mtime) at both read points:
1. `analysis/postprocess/run_mva_postprocess.py` — `process_era()`'s
   `era_input.glob('**/*.parquet')`.
2. `combine/scripts/create_datacards_ctag2d_100150.py` — `load_sumw()`,
   which independently re-globbed the same tree for the denominator (both
   numerator and denominator needed the fix, or dedup would have been
   inconsistent between them).

Verified the fix collapses `2022postEE` HPlusCharm/ZZto4L to exactly 468/208
files. Rebuilt: full rescoring, all 4 eras, into new
`hplusc_mva_4class_ctag2d_scored_v5` (`_v4` left in place, untouched, as the
known-buggy reference — nothing deleted). New datacard rates land almost
exactly on the *old* 08-28 numbers (Signal 0.0538, ggZZ 2.7335, qqZZ
36.3309, Other_Higgs 56.2245), consistent with `_v2`/`_v3` (used for the
301.0 result) having accumulated far less duplication than `_v4` by the
time it was built.

**Combine result, reproduced bit-identically on 2 runs**:

| CL | r | kappa_c |
|---|---|---|
| 2.5% | 153.26 | 29.21 |
| 16.0% | 209.24 | 38.94 |
| **50.0% (median)** | **299.50** | **54.60** |
| 84.0% | 439.18 | 78.82 |
| 97.5% | 621.54 | 110.42 |

**r=299.5/kappa_c=54.60 was, as of 2026-08-30, the trustworthy production
reference** — essentially identical to the pre-"fix" r=301.0/kappa_c=54.86
(0.5% difference, noise-level). **Superseded 2026-08-31 by the ctag2d
3-jet-cap fix's reprocessing: current reference is r=299.0/kappa_c=54.52**
(negligible 0.17% shift from this number) — see "Update 2026-08-31 (later)"
near the end of this file. The `lhescale.py` Up/Down-swap and gg-lnN split
are still correct fixes worth keeping, but once duplication is removed they
produced **no meaningful net change** to the limit — the entire apparent
301.0→219.0 improvement was the duplication bug.

Outputs: `combine/outputs/combine_run3_100150_ctag2d_v5/` (datacard, ROOT
histograms). `--scored-dir` default in `create_datacards_ctag2d_100150.py`
should be updated from `_scored_v4` to `_scored_v5` before any further use
of that script (not yet done — flagged for the next session touching this
file).

**How to apply going forward**: (1) **any future read of the raw ctag2d
postprocess tree must go through `dedupe_parquet_files_by_source()` or an
equivalent GUID-based dedup** — a bare glob against
`outputs/hplusc_mva_4class_ctag2d/` will double/triple-count. (2) A
same-file full-syst-vs-stat-only sanity check (`-S 0` or
`--freezeParameters allConstrainedNuisances`) is now a standing
recommended step before trusting any new production number — it's what
caught this. (3) Bit-identical reproducibility across repeated `combine`
runs proves a result is *stable*, not that it's *correct* — a bug in the
input data reproduces exactly as reliably as a correct result does.
`[90,160]`/with-ZX and `_3ratio` have **not** been rebuilt against
`_scored_v5` yet. Full writeup, including the diagnostic chain that found
this: `second-brain/Tasks/HcZZ-fake-rate.md`, "Update 2026-08-30 (later)".

---

### Update 2026-08-30 (even later): was r=301.0 itself clean? No — same bug
in `_scored_v3`; impact ranking rebuilt on `_v5`

Follow-up question: does the *old* production (`_scored_v3`, which fed the
r=301.0 result) carry the same duplication? Checked directly — **yes**,
concentrated in `ZZto4L`/qqZZ:

| Era | HPlusCharm (Signal) | ZZto4L (qqZZ) |
|---|---|---|
| 2022preEE | 4 files / 2 GUIDs (tiny N) | 190 / 82 = **2.3×** |
| 2022postEE | 475 / 468 = 1.5% (negligible) | 666 / 208 = **3.2×** |
| 2023preBPix | 435 / 435 = clean | 332 / 165 = **2.0×** |
| 2023postBPix | 3 files / 1 GUID (tiny N) | 170 / 81 = **2.1×** |

qqZZ — the dominant background — was duplicated 2.0–3.2× in **every era**
of `_v3`, already present at the 2026-08-27/28 build. Signal/HPlusCharm's
duplication in `_v3` was small or based on too few files to matter.
**r=301.0 was never a clean baseline** — its near-agreement with the
properly-deduped r=299.5 looks like coincidence (how a duplicated
background vs. a duplicated signal each push the fit, non-linearly, don't
have to cancel the same way), not evidence `_v3` was unbiased. `_v5`
(deduped from the raw tree, all 4 eras, verified against unique-GUID
counts) is the **only** number in this project's history confirmed built
from a non-duplicated source tree.

**Impact ranking rebuilt on `_v5`**: top-2 unchanged —
`CMS_eff_e_reco_below20` (−4.01%, was −4.57% on the buggy `_v4` card),
`CMS_ctag2d` (−3.51%, was −3.88%). `CMS_eff_m_id` confirmed exactly zero
impact again, now across every version of this card. Full ranked deltas (r
units): `CMS_eff_e_reco_below20` −12.0 (−4.01%), `CMS_ctag2d` −10.5
(−3.51%), `BR_HZZ4l` −2.0 (−0.67%), `lumi_Run3` −1.0 (−0.33%),
`QCDscale_gg`/`pdf_gg`/`ps_fsr`/`CMS_pileup`/`CMS_eff_e_reco_20to75` each
−0.5 (−0.17%), remaining 8 systematics exactly 0.00. Output:
`combine_run3_100150_ctag2d_v5/impact_ranking.json`/`.png`, also copied to
`second-brain/slides/AN_HcZZ_figures/hczz_ctag2d_100150_v5_impact_ranking.png`
and wired into a new slide in `combine_summary_hczz.tex`.

Full writeup: `second-brain/Tasks/HcZZ-fake-rate.md`, "Update 2026-08-30
(even later)".

---

### Update 2026-08-31: `CMS_ctag2d`'s 143.6 impact investigated — 3-jet cap
replaces the unbounded all-`selected_jets` product

Flagged the 143.6 real-Impacts value (previous update) as suspiciously
large directly. The 2026-08-18 "documented caveat, keep the scope" decision
(Known traps, above) was made against the coarse proxy's 71.2 — roughly half
of what the real profile-likelihood fit later found — so it was revisited
rather than assumed still valid.

**Root cause quantified**: queried `flavTaggingSF_2022postEE.json.gz`
directly across the full (flavor∈{0,4,5}, all 11 WP codes, pt 20–500 step 10)
grid — **648/1584 (41%) return the exact placeholder triple
`central=1.000, up_Total=3.000, down_Total=0.300`**, not a rare corner.
`CTag2DCorrector` took `ak.prod` over every jet in `events.selected_jets`
(pt>20/|η|<2.5/tight jetID+lepton veto/ΔR>0.4 — **uncapped multiplicity**),
so events with 2 independent placeholder-corner jets compound to 3.0×3.0=9.0
(or 0.3×0.3=0.09). Verified on real qqZZ `_scored_v5` parquets (692 events,
2022postEE): mean(Up/nom)=1.066, **max=9.0**; mean(Down/nom)=1.004 (looks
flat only because the tail is two-sided and cancels on average, not because
Down doesn't vary), **min=0.09**. 1.5% of events have Up/nom>5.

**Checked HWW's scope directly** (cloned `Chirayu18/higgscharm`,
`hww-analysis` branch, read-only): their `CTag2DCorrector` applies the SF to
exactly **one** jet (`ak.firsts(events.selected_candidate_cjet)`, their
MVA's single leading-CvsL jet) — no `ak.prod`. **Not a fix to copy**: our
`models/config_4class.yml` has `n_cpf_candidates: 3` with an explicit design
comment that the model deliberately needs **all jet flavors** (b-jets for
H+b, light-jets for VBF/qqZZ), not just c-jets — matching HWW's 1-jet scope
would silently stop correcting jets 2–3's real tagging-SF mismodeling, which
our classifier actually consumes.

**Jet-multiplicity check** (`jet_multiplicity`, all 4 eras, `_scored_v5`):

| Process | 1 jet | 2 jets | 3 jets | >3 jets | total | frac >3 |
|---|---|---|---|---|---|---|
| Signal | 32269 | 6808 | 1042 | 190 | 40309 | 0.47% |
| qqZZ | 237911 | 74110 | 17645 | 4547 | 334213 | 1.36% |
| ggZZ | 425938 | 143606 | 31745 | 6806 | 608095 | 1.12% |
| Other_Higgs | 359480 | 268887 | 108656 | 60881 | 797904 | 7.63% |

Most events already have ≤3 jets (the 9.0× qqZZ example above is a ≤3-jet
process) — a 3-jet cap removes the unbounded tail from 4+-jet events (mostly
Other_Higgs, via ttH/VBF) but does **not** eliminate the core
placeholder-compounding pathology. Documented honestly as a partial,
physically-motivated mitigation (matches the MVA's real jet dependency), not
a full resolution — excluding placeholder cells from the uncertainty
entirely would be the deeper fix, deferred (not requested this session).

**Fix implemented** (EOS + AFS, kept in sync — AFS was also found missing
the entire 2026-08-18 docstring paragraph above, a pre-existing
checkout-drift gap, backfilled while here): `analysis/corrections/
ctag2d.py`, `CTag2DCorrector.__init__`: `jets = events.selected_jets` →
`jets = events.selected_jets[:, :3]`. Awkward's per-sublist slicing
truncates each event's (pT-descending, native NanoAOD order) jet list to at
most 3, leaves shorter lists untouched — verified directly
(`ak.Array([[1,2,3,4,5],[10,20],[],[100]])[:, :3]` → `[[1,2,3],[10,20],[],
[100]]`). Both checkouts `py_compile`-clean; `diff` shows only the intentional
backfill sentence. Full coffea-level smoke test blocked by a pre-existing,
unrelated numpy/numba version conflict in AFS's `.local` site-packages
(documented standing issue) — relied on the isolated awkward unit check
instead.

**Not yet done**: rebuild `_scored_v6`/rerun combine with the capped
corrector to get the corrected r/impact numbers (needs the same condor
reprocessing pass already planned for the `lhe_alphaS` fix — not run this
session). Full writeup: `second-brain/Tasks/HcZZ-fake-rate.md`, "Update
2026-08-31"; memory `hczz_ctag2d_impact_investigation`.

### Update 2026-08-31 (later): reprocessed end-to-end — 3-jet cap confirmed
a partial mitigation only, `CMS_ctag2d` impact essentially unchanged

Full condor reprocessing (`hplusc_mva_4class_ctag2d`, MC-only, all 4 eras,
84 (era, dataset) pairs from the workflow's own `mc`+`signal` keys —
80/84 succeeded, only the known-gone `bbH_Hto2Zto4L` failed everywhere;
3 resubmit rounds closed the real `cms-xrd-global.cern.ch` XRootD-timeout
gaps on datacard-relevant datasets, mostly `ZZto4L`/qqZZ) → rescore
(`run_mva_postprocess.py --all-eras --no-mass-window` into
`hplusc_mva_4class_ctag2d_scored_v6`, **0 errors**, 1,847,922 events) →
datacard+combine rebuild (`run_pipeline.sh --merge-bin-ranges "2-4,7-10"
--skip-zx`, `--scored-dir` pointed at `_v6`).

**Result: median r = 299.0, kappa_c = 54.52** (was 299.5/54.60 — a 0.17%
shift, statistically negligible). **Real impact ranking (`impact_ranking.py
--real --expect-signal 299.0`): `CMS_ctag2d` still clearly #1, impact =
142.17** (was 143.63 — ~1% shift). Confirms the prediction made when
scoping the fix: the 3-jet cap only bounds the 4+-jet tail; it does not
remove the core placeholder-SF-compounding pathology, since 2 of an
event's remaining ≤3 jets can still both land in the 41%-placeholder
region. The fix is correct and worth keeping, but the deeper fix
(excluding placeholder SF cells from Up/Down evaluation entirely) is
still the only known way to actually move this nuisance's impact — not
implemented.

**Current reference now**: r=299.0/kappa_c=54.52,
`hplusc_mva_4class_ctag2d_scored_v6`, `combine/outputs/
combine_run3_100150_ctag2d_v6`. Full writeup: `second-brain/
Tasks/HcZZ-fake-rate.md`, "Update 2026-08-31 (later)"; memory
`hczz_ctag2d_impact_investigation` (RESOLVED section). Slides:
`second-brain/slides/hczz_wg_meeting_20260831.tex`.

### Update 2026-09-02: `CMS_ctag2d` up_Total/down_Total swap for WP=L0
fixed — the real driver of the impact anomaly, not the 3-jet cap above

A 2026-09-01 session (context lost before write-up, recovered via
`git status` at the start of this session — **lesson: check `git status`
on both checkouts before assuming the Tasks log reflects the current
working tree**) found `up_Total`/`down_Total` physically swapped in
`flavTaggingSF_<era>.json.gz` for WP=L0 ("fails the tag"): reading the
raw correctionlib content directly, `up_Total < central < down_Total` for
~18/198 (flavor,wp,pt) cells, 100% WP=L0, 0/198 anywhere else. Distinct
from the 41%-corner exact placeholder triple above (central isn't 1.0
here) — a separate, isolated bug. Fixed in
`analysis/corrections/ctag2d.py` (`_eval_flat_updown_corrected()`, both
checkouts): detects the swap generically per-jet (`up_flat < down_flat`),
self-corrects against any future SF-file version. Also wired
`weight_higgs_plus_c` into `create_datacards_ctag2d_100150.py` for the
first time (present in parquets since `higgsHFWeight: true` was enabled
2026-08-28, never used until now).

Reprocessed all 4 eras (`hplusc_mva_4class_ctag2d_scored_v7`, 0 errors,
1,847,820 events) and rebuilt the `[100,150]` no-ZX card:

**Result: median r = 413.0, kappa_c = 74.28** (was 299.0/54.52 — a 38%
increase). **Real impact ranking (`impact_ranking.py --real
--expect-signal 413.0`): `CMS_ctag2d` impact = 195.81** (was 142.17), now
~11.7× the #2 nuisance (was ~3.2×) — 18 systematics total,
`weight_higgs_plus_c` new at #11 (impact=7.51). Central yields verified
essentially unchanged (Signal 0.053778→0.053781, qqZZ/ggZZ/Other_Higgs
likewise flat) — this is purely a systematics-shape effect, not a
yield/rate change.

**Pushed back hard on the jump** — verified three independent ways
before accepting it: (1) stat-only limit (all 18 nuisances frozen)
identical at r=269.0 before and after — proves 100% of the delta is
`CMS_ctag2d`'s shape, zero yield/duplication component; (2) independent
jet-level recomputation straight off the raw calibration file (bypassing
the pipeline's stored weight columns entirely) on 2183 real Signal
events reproduced a ~4× widening of the per-event Up/Down spread from
scratch; (3) cross-checked against the file's own separately-reported
`up_Stat`/`down_Stat` (a component this analysis doesn't even read) —
those follow the normal `up>central>down` convention on the exact same
L0 cells where Total is inverted, and `up_Total` is bit-exact identical
to `down_Stat` in most cells checked, corroborating the swap diagnosis
from an independent, unused quantity in the same file.

**Caveat, stated explicitly**: this is well-supported by internal file
consistency and independent verification, but **not confirmed by
whoever produced this preliminary `2D_HF_Tagging` calibration file** —
treat as the best current understanding, not an external sign-off. The
still-open 41%-corner placeholder triple (untouched by this fix) also
contributes to `CMS_ctag2d`'s remaining large uncertainty.

**Current reference now**: r=413.0/kappa_c=74.28, `CMS_ctag2d`
impact=195.81, `hplusc_mva_4class_ctag2d_scored_v7`, `combine/outputs/
combine_run3_100150_ctag2d_v7`. Full writeup, all verification detail:
second-brain memory `hczz_ctag2d_l0_updown_swap_fix`; `second-brain/
Tasks/HcZZ-fake-rate.md`, "Update 2026-09-02" entries. Plot:
`second-brain/slides/hczz_overview_ctag2d_sf_band.pdf` (pre/post-fix
bands shown side by side).

**Still open**: no POG confirmation (see caveat above); `[90,160]`/
with-ZX/`_3ratio` variants not rebuilt against `_v7`; the deeper
placeholder-triple fix (excluding those exact cells from Up/Down
entirely) remains unimplemented. **Checked and closed**: the 13
(era,dataset) pairs flagged as residual-risk from this campaign's
`.err`-traceback scan were re-checked against `jobs_status.py`'s real
completeness logic — all 13 are fully complete; the only real gaps
(22 jobs total, 4 eras) are non-mapped MC or data streams, irrelevant to
this datacard. `_v7` is confirmed complete for every process it uses.

---

## Known traps

| trap | detail |
|---|---|
| **No merge step (historical)** | §2 above describes the old split path — superseded 2026-08-14 for new workflows by the consolidated `--postprocess --mva-inference` flow, which had never been run successfully before that date. |
| **Local `apptainer exec` smoke tests need explicit binds, not automatic** | Found 2026-08-19 running a local single-file smoke test outside condor: `/eos`, `/afs`, `/cvmfs`, and the Kerberos ticket cache dir (`/run/user/<uid>`) are **not** auto-mounted into the container in this environment the way they are on a real condor worker / lxplus node. Symptoms in order as each missing bind was found: `/eos` missing → `FileNotFoundError` on `submit.py` itself; ticket cache missing → EOS `Permission denied` even with `/eos` bound (klist inside the container shows no cache until `/run/user/<uid>` is also bound); `/cvmfs` missing → correctionlib files under `/cvmfs/cms-griddata.cern.ch/...` not found even though the image itself lives on `/cvmfs` (loading the image doesn't imply the whole `/cvmfs` tree is bound). Full working invocation: `apptainer exec --bind /eos --bind /afs --bind /run/user/<uid> --bind /cvmfs <image> bash -c 'export HOME=<scratch>; export PYTHONNOUSERSITE=1; export X509_USER_PROXY=<path to a valid voms proxy>; ...'` — also needs a genuinely valid (non-expired) X509 proxy for xrootd reads (Kerberos alone isn't enough for CMS xrootd doors), separate from the EOS-filesystem Kerberos requirement above. |
| **`merge_parquets()` required identical schemas, crashed on real data** | Fixed 2026-08-14: rewrote to `pa.concat_tables(tables, promote=True)` (schema union) instead of `dask.dataframe.read_parquet(glob).compute()` (strict schema). Real mismatch found in `EGamma0v1D`/`DYJetsToLL_50` (data/DY, not analysis processes) — some files predate the workflow yaml adding `nSV`/`jet_btagUParTAK4*`/`jet_h_dphi_inclusive`. Signal/qqZZ/ggZZ/Other_Higgs directories confirmed schema-clean. Also fixed: empty-partition crash (`pa.concat_tables([])`) for datasets with zero legitimate rows. |
| **Unregistered private datasets crash multiple postprocess functions** | `2023postBPixHB`-style private datasets (condor-run directly, never added to `<era>_nanov<n>.yaml`) crash `filesets/utils.py::get_process_sample_map`, `postprocess/postprocessor.py::fill_histograms_from_parquets`, and `postprocess/postprocessor.py::save_histograms_by_sample`'s `get_lumi_weight` call, all with a bare `KeyError`. All three fixed 2026-08-14 to skip/degrade gracefully instead of crashing (the last one skips the whole sample's histogram build — no safe default xsec exists). A fourth instance, in the final cutflow-summary print (`key_process_map[key]` for `SomeSMSignal`), has no code fix — **use `--nocutflow`** (see below) instead, since that whole block is optional reporting output, not something `--mva-inference` needs. The real underlying gap (these datasets were never registered at all) is still open. |
| **`--nocutflow` required for `hplusc_mva_4class` + `--mva-inference`** | Without it, `run_postprocess.py` crashes in the final cutflow-summary print (see above) one step before MVA inference ever starts. Always include it for this workflow. |
| **Use `models/best_model.pt` + `models/hc_zzto4l_mw_training_4class_nomass.yml`, not a `b-hive_ttcc/output/TrainingTask/.../best_model.pt` path** | These two files are the DOCUMENTED (`models/README.md`), checkpoint-bundled pair for `run_postprocess.py --mva-inference`. The `.pt` is byte-identical (MD5-confirmed) to `train_loose_oldcwithmassbal/best_model.pt`, dated 2026-03-23 — same model, not a different one — but `b-hive_ttcc/config/hc_zzto4l_mw_training_4class_nomass.yml` (same filename, different file) had drifted to 103 `global_features`. An earlier same-day fix there (commenting out `jet_ht`/`nSV`) hit the right COUNT (101) by coincidence but the WRONG COMPOSITION (kept `jet_h_dphi_inclusive`, dropped `jet_ht` — the correct config is the reverse). Matching total dimension is NOT sufficient evidence a feature list is correct — diff the actual `global_features` names against a known-good config. Both `.yml` files now correctly match (fixed 2026-08-14) — see `hczz_mva_postprocessor_root_cause` memory for the full diagnostic trail (culminating in a full-population 71.04% argmax accuracy check on real data, exceeding the model's own native ground-truth benchmark of 69.64%). |
| **Two real code bugs in `mva_inference.py`, fixed 2026-08-14, unrelated to the config issue above** | (1) `MVA_JET_TAG_BOUNDARIES` was a single fixed (wrong, "pre-official-calibration") boundary set for the `jet_is_*` one-hot features (33/101 inputs) — now era-aware (`JET_TAG_BOUNDARIES_BY_ERA`, same real per-era boundaries validated for `ctag2d.py`), threaded via a new `MVAPostProcessor(era=...)` kwarg; `run_mva_postprocess.py`/`run_zx_inference.py` reuse one processor across a per-era loop so each needed `processor.era = era` set inside the loop, not just at construction. (2) per-candidate-SPECIFIC columns (`z1_l2_pt`, `z2_pt`, etc. — each a length-1 array for one specific lepton/Z role) were read with `idx=j` (the candidate slot), only correct for `j=0`; slots 1-3 silently zeroed. Fixed via a new `_get_candidate_col()` helper that reads specific-name columns at a fixed `idx=0`. Neither bug was the dominant cause of the collapse symptom (re-tested with both fixed, score collapse barely changed) — the config composition bug above was. |
| **`ZZto4L` / `GluGluHtoZZto4L` substring collision** | `"ZZto4L" in "GluGluHtoZZto4L"` — naive dict-order matching silently misrouted every `GluGluHtoZZto4L` event into `qqZZ` for an entire session. Fixed 2026-08-04 by matching the longest key first in `get_process_from_path()`; the fix is now load-bearing, don't reorder `PROCESS_MAPPING` casually. |
| **sumw from parquet shard metadata** | Same failure mode `hww-analysis` warns about (undercounts low-efficiency samples). Fixed here via `sumw/*.json` sidecars (§1) — `load_sumw()` raises rather than falling back if they're missing, don't defeat that. |
| **`CMS_ctag_light` sign flips (historical, retired 2026-08-14)** | `ctag.py`'s untagged-jet formula `(1-SF*eff)/(1-eff)` goes negative when `SF*eff>1` — hits the light-flavor mistag SF's up variation in some pT/η bins. Confirmed genuine sign flips in every process's existing 1D-scored parquets. **Clamped at the source, not truly fixed** — a numerical safety net (BTV-POG-standard mitigation) so the code doesn't crash, not a resolution of the underlying disagreement; the clamp is exactly the kind of thing this project's standing rule rejects as "fixed" (see the earlier ratio-clip incident). This is why `CMS_ctag_light` stayed excluded from the datacard even after clamping, right up until production switched its whole SF scheme to `ctagging_2d` (which has no `eff/(1-eff)` formula at all, so this failure mode is structurally impossible, not just clamped around). `hww-analysis`'s `ctag.py` has the identical unclamped formula, still live in their repo — irrelevant to their own results only because their production never calls it (`ctagging_2d` there too), same as ours going forward. |
| **Condor rejects `/eos` paths in submit files (CERN batch policy change since ~March)** | `condor/submit.sub` + `submit_condor.py`'s `CONDORDIR`/`LOGDIR` substitution point `executable`/`output`/`error`/`log` at literal `/eos/...` paths — confirmed working from this exact repo in March 2026 (found a successful old `.log`), now rejected outright: `ERROR: Failed to commit job submission into the queue... Standard batch schedds cannot use /eos paths directly`. **Workaround confirmed 2026-08-14: submit from the AFS checkout instead** (`/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm/`, same `runner.py -w <workflow> -y <era> --submit --eos --output_format parquet`) — condor accepts submit files whose paths originate from AFS even though `--eos` still points job *output* at `/eos`; no code change needed, just run from a different filesystem. Not a permanent code fix (submit.sub itself still hardcodes `/eos` if invoked from an EOS cwd) — still open whether to patch `submit.sub` properly (xrootd URLs / EOS-submit schedds) or keep "submit from AFS" as the accepted permanent workflow. |
| **`lhe_pdf` formula: Hessian vs. MC-replica (RESOLVED 2026-08-03, root cause of residual size confirmed 2026-08-28)** | Fixed 2026-08-03: formula now divides by `N-1=99` (`sqrt(Σ(w_k-w0)²/99)`), correct for the NNPDF Monte-Carlo-replica prescription our private samples use. The private NanoAODs were regenerated (`hplusc_htozz1`/`hplusb_htozz1`, reachable again) and read directly 2026-08-28 — the formula is not the remaining problem. The residual ~90-130% `delta_pdf` for Signal/HPlusBottom is real and traced to the samples' `MuRFScaleDynX0p50` central-scale choice (half a dynamical reference scale), confirmed from raw `LHEScaleWeight`/`LHEPdfWeight` branches, not a formula bug — see the 2026-08-28 Systematics update above. |
| **Signal/HPlusBottom negative-weight fraction is ~25%, not incidental** | Measured directly from `hplusc_mva_4class_scored_v4` (2026-08-14): `HPlusCharm`/`HPlusBottom` run **24–26% negative `weight_nominal`** across every era, roughly **8–17× higher** than every other process (`ZZto4L` 0.18%, `GluGlu*` ggZZ ~0.003–0.01%, Higgs backgrounds `WH`/`ZH`/`TTH` 1.4–1.7%). This is the same two processes with the already-diagnosed `lhe_pdf`/`scalevar_muR/muF` blowups — plausibly the same root cause (private amc@nlo-based generation), not yet connected. See [Open items](#open-items--next-techniques-to-port-from-hww-analysis). |
| **Central DY NanoAODv12 datasets are INVALID in DBS (found 2026-09-23)** | Every central DY query in `analysis/filesets/*_nanov12.yaml` (all 4 Run3 eras) plus `bbH_Hto2Zto4L` is `INVALID` in global DBS. The chains were re-made from DIGI-RECO in 2025-09 to 2025-12, so the old files are disappearing from disk. The symptom is jobs failing within about 2 min in preprocessing with `[3011] No servers are available to read the file`. `jobs_status.py`'s site blacklist can't help, and a plain resubmit fails the same way. Check with `dasgoclient -query="dataset dataset=<DS> status=*"`. Plain `file dataset=<INVALID DS>` returns 0 files. The replacement map is in second-brain `Notes/hczz-dy-nanov12-replacement-map-2026-09-23.tsv`. **Only `hplusc_mva_4class_CR_merged` has been switched** (at the condor job-folder level); the shared yamls/filesets still point at INVALID DY. |
| **`condor/submit.sh` exits 0 even when `xrdcp` of the output failed** | The script ends with `rm -rf "$LOCAL_OUTPUT"`, whose exit code becomes the job's. On 2026-09-23 about 180 jobs failed `xrdcp` with `[3021] ... Disk quota exceeded` and many still reported `return value 0`. Partially copied jobs can also look complete to `jobs_status.py`. After any copy/quota incident, grep the `.err` files for `xrdcp`/`[3021]`, and wipe and rerun the affected datasets instead of trusting condor exit codes. Not yet fixed in code. |
| **EOS user quota: the file-count limit (5M) runs out before bytes (2 TB)** | `/eos/user/s/snandaku` hit 5.00M/5.00M files at 93% bytes on 2026-09-23. Check with `EOS_MGM_URL=root://eosuser.cern.ch eos quota /eos/user/s/snandaku/` (without `EOS_MGM_URL` it says "no quota for path"). Parquet outputs are many small files (~220 per CR_merged job). `eos find --count` stops at ~100k per call. `eos rm` goes to the recycle bin, which does not count against the user quota. |

---

## Open items / next techniques to port from `hww-analysis`

**Priority order re-ranked 2026-08-21** after a systematics-completeness audit vs Felix
Heyen's thesis / HIG-24-013 / the HWW deck (full writeup: second-brain memory
`hczz_systematics_completeness_audit`, `Tasks/HcZZ-fake-rate.md` "Update 2026-08-21"). New
#0: **muon efficiency SF (ID/ISO/trigger) is entirely unapplied for ZZ→4l** — not a missing
uncertainty like the items below, the *central value* itself is missing (no `muon:` key in
any ZZ→4l workflow yaml, confirmed via `grep -rn "^\s*muon:" analysis/workflows/*.yaml`;
`MuonWeights.add_id_weights()`/`add_iso_weights()` in `analysis/corrections/muon.py` already
work, just never invoked here). Felix's thesis quotes the equivalent `muonSF` nuisance at
~4.5–9% yield effect — since 3 of 4 ZZ→4l final states carry muons, this is a plausible
few-percent normalization bias on every HcZZ number on record, not just an uncertainty gap.
Needs a `muon:` key added to the workflow yaml plus a full reprocessing — blocked behind the
same condor `/eos`-path issue as items 3–4 below. Electron ID SF has the same class of gap
(`id: false`, documented/deliberate, smaller likely magnitude). Also newly flagged: a
dedicated FS (flavour-scheme, 4FS/5FS envelope) uncertainty for Signal/HPlusBottom — Felix's
thesis has one (~59%/42% yield effect, his "clearly leading" systematic), HcZZ has none; a
candidate *addition*, not a fix for the already-excluded `lhe_pdf`/`scalevar` rows (different
problem). `pdf_qq` was checked and resolved as **not** a gap — qqZZ's real `lhe_pdf` shape
systematic already covers it.

Original priority order set 2026-08-14; update as items land.

1. **`ctag2d` pseudo-continuous SF migration — done, active in production
   as of 2026-08-17** (18-systematic datacard pipeline as of 2026-09-02 for
   `[100,150]` no-ZX; `[90,160]`/with-ZX not yet rebuilt against the same
   count — see the `ctag2d` variant subsection under Systematics above).
   `analysis/corrections/ctag2d.py`
   ported from `Chirayu18/higgscharm`'s `hww-analysis` branch (same
   `L0=0, C0..C4=40..44, B0..B4=50..54` WP-code mapping, `(syst, flavor, wp,
   abseta, pt)` evaluate order) — this resolved the CERN-SSO-gated-doc
   blocker in `hczz_pnet_ctag_sf_migration` memory outright, no guessing
   needed. **Directly verified against our real files** (2026-08-14,
   `flavTaggingSF_2022postEE.json.gz`): raw schema matches exactly, all WP
   codes evaluate without error, `up_Total`/`down_Total` give distinct sane
   values. A synthetic multi-jet smoke test of the full `CTag2DCorrector`
   class (0/1/2-jet events, product-across-jets combination) also passed.
   The file additionally exposes a full per-source decorrelation structure
   (`up/down_Stat_flav{L,C,B}_<cat>_pt_<bin>`, `up/down_JES`, `up/down_PUWeight`,
   etc.) beyond the single `Total` nuisance currently wired — worth revisiting
   once the coarse version is validated in a real datacard.

   **Scope decision made during porting**: `hww-analysis` applies this to one
   "candidate c-jet"; this port instead multiplies a per-jet SF across every
   jet in `events.selected_jets` (`ak.prod`), matching the scope of the 1D
   `CTagCorrector` it's meant to replace. **Reviewed and confirmed 2026-08-18**
   after root-causing the CMS_ctag2d Down-variation asymmetry to this scope (see
   Known traps above) — decided to keep all-jets, not a bug.

   **Update 2026-08-14 (later): comparison workflow + datacard variant
   built.** `analysis/workflows/hplusc_mva_4class_ctag2d.yaml` (both
   checkouts) — byte-identical to production except `event_weights` swaps
   `ctagging: [wp: loose]` for `ctagging_2d: true`. The `atleast_one_cjet`
   **selection cut** (`working_points.jet_ctagging(events, 'loose', year)`)
   is deliberately untouched — that's a separate decision from the SF
   scheme (see "selection vs. SF" note below) and changing it would
   invalidate the MVA training + every reference number, not just add a
   comparison point. `b-hive_ttcc/combine/create_datacards_ctag2d.py` (copy
   of `create_datacards.py`, `CMS_ctag_b`/`CMS_ctag_c` swapped for a single
   `CMS_ctag2d` nuisance, default paths pointed at the ctag2d workflow's own
   output tree so it can never collide with production).

   **Offline yield-impact check, real data, no reprocessing needed**: since
   `CTag2DCorrector` only needs `jet_pt/eta/btagPNetCvL/btagPNetCvB/
   hadronFlavour` — already stored in existing production parquets as the
   *full* selected-jet collection, not just c-tagged jets — ran it directly
   against those stored columns and compared to the existing
   `weight_CMS_ctag_b × weight_CMS_ctag_c × weight_CMS_ctag_light` product,
   without touching NanoAOD/condor at all. Single-file check (10 events,
   `GluGlutoContinto2Zto4Tau_1`/2023postBPix): new/old yield ratio 0.942.
   Full-statistics version (all 4 eras, all 4 processes) run 2026-08-14 —
   see the "Offline ctag2d yield comparison" note below for the result.
   This offline check is a rough sanity read only — it reuses the OLD
   1D-scheme's selection (whatever jets already passed `loose` c-tag in the
   stored parquets), not a true reprocessing under the ctag2d workflow, so
   treat it as directional, not a substitute for a real campaign.

   **BUG FOUND AND FIXED 2026-08-14 (later): category boundaries were wrong.**
   The edges ported from `hww-analysis` (`[0.0,0.250,0.452,0.808,1.0]` /
   `[0.0,0.006,0.017,0.055,0.761,0.944,0.985,0.995,1.0]`) turned out to be the
   **2024 UParT** boundaries, not 2022/2023 PNet — caught after pulling up
   the real `etsai.web.cern.ch` doc page (every entry there is
   explicitly "Tagger: UParT v2, NanoAODv15", files under `phys_top/Run3Vcb/`,
   not our `phys_higgs/cmshgg/` PNet files) and the actual per-era
   `HPC_ctag_WPs` JSON, which matched exactly what was already correctly
   implemented (independently, for a different purpose — MVA training
   features) in `b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::
   JET_TAG_BOUNDARIES_BY_ERA`. `ctag2d.py` now has real per-era boundaries
   (`BOUNDARIES_BY_ERA`, 4 PNet eras) copied from that confirmed-correct
   source; `_category_np()` takes an `era` argument. Re-validated: clean
   tiling of the full plane (0 unassigned points, 500×500 grid, all 4 eras),
   all 11 WP codes still evaluate. **The coordinate transform itself was
   never wrong** — only the edge values.

   **Corrected full-statistics offline comparison** (all 4 eras × all 4
   processes, re-run 2026-08-14 after the fix):

   | Process | Old yield (1D SF) | New yield (2D SF) | Ratio | Buggy-version ratio |
   |---|---|---|---|---|
   | Signal | 17,382.1 | 17,840.0 | **+2.63%** | +2.67% |
   | qqZZ | 1,082,525.6 | 1,083,540.3 | **+0.09%** | −3.40% |
   | ggZZ | 841,401.4 | 851,363.0 | **+1.18%** | −3.67% |
   | Other_Higgs | 3,271,437.1 | 3,278,789.6 | **+0.22%** | −2.33% |

   The corrected picture is much more modest than the buggy version
   suggested: Signal is still up ~2.6%, but backgrounds are close to flat
   (qqZZ +0.09%, Other_Higgs +0.22%) or even slightly up (ggZZ +1.18%) — not
   down 2-4% each like the wrong boundaries implied. **Do not cite the
   buggy-version ratios for anything** — they're left in the table only to
   show how much a wrong category mapping can distort a yield check that
   otherwise runs and completes without any error.

   **Reciprocal finding**: `hww-analysis`'s own `ctag2d.py` uses this same
   single fixed (2024 UParT) edge set across all 4 of *their* PNet eras too
   — their production analysis likely carries the identical bug. Worth
   flagging to that repo's maintainer.

   **Selection vs. SF, clarified**: the 2D file's `wp` categories
   (`L0`/`C0`-`C4`/`B0`-`B4`) are not a selectable "loose"-equivalent working
   point the way the 1D tagger has — they're the category grid the SF is
   binned in. Using them to redefine the **selection** cut (not just the SF)
   is a separate, bigger decision (different signal acceptance, needs its
   own purity/efficiency study, invalidates current training) — not
   something to bundle into this SF-only migration.

   **Wired but inert in production**: `correction_manager.py` (both EOS +
   AFS) has a `ctagging_2d` branch, gated behind `event_weights.ctagging_2d:
   true` — production `hplusc_mva_4class.yaml` still has only `ctagging:
   [wp: loose]`, untouched. Turning both on in the same workflow would
   double-apply c-tag SFs (both correctors touch the same jets) — this is
   exactly why the comparison lives in a separate `_ctag2d` workflow file
   rather than a flag flip on the production one.
   Also directly retires the `CMS_ctag_light` exclusion above (2D SF
   doesn't use the broken eff/(1-eff) untagged formula) — but only once
   parquets are reprocessed with it enabled.
2. **negrw ML reweighting — evaluate, likely needed.** `hww-analysis` built
   this for V+jets; we have no V+jets background, but the 25% negative-weight
   finding above means the *same class of problem* (NLO-generator
   negative-weight variance blowing up shape systematics) shows up in
   `Signal`/`HPlusBottom` instead. Worth trying before assuming it's
   unnecessary — this is the leading unresolved-bug candidate for the
   excluded `lhe_pdf`/`scalevar_muR/muF` rows in `USABLE_SYST`.
3. **`autoMCStats`** — **added to the datacard 2026-08-18** (threshold 10).
   Update 2026-08 (separate investigation, `ctag2d` pipeline): found
   numerically degenerate with the current binning (several bins have MC-stat
   *errors* 10–90x the bin *content*). **Update 2026-08-19**: threshold made a
   CLI flag (`--automcstats-threshold`) to test raising it past this
   degeneracy — see the dated update in the Systematics section above for the
   full rationale and in-progress result.
4. **Object-shift (JES/JER/lepton scale+res) systematics** — bigger lift,
   needs new shift-directory production in `runner.py` plus inference
   re-coverage (§4). **Update 2026-08-17**: electron reco is actually *not*
   blocked — `weight_CMS_eff_e_reco_{20to75,above75,below20}_<year>Up/Down`
   already exist in the `ctag2d` scored parquets (presumably from an earlier,
   unrelated production step) and are now wired into `create_datacards_
   ctag2d*.py`'s `USABLE_SYST` (16 systematics total, see the `ctag2d`
   variant subsection under Systematics above). **Update 2026-08-19**:
   JES/JER (`CMS_scale_j`/`CMS_res_j`) and muon scale/resolution
   (`CMS_scale_m`/`CMS_res_m`) shift *production* is now code-complete and
   smoke-tested (`jec_shifts` workflow, `hplusc_mva_4class_ctag2d_
   jecshifts.yaml`) — see the dated update in the Systematics section above.
   Building a datacard that actually reads these shift directories as shape
   systematics is separate, not-yet-started work. Muon ID/isolation/reco
   (efficiency SF, different from scale/resolution) remains genuinely
   blocked — no such weight columns exist anywhere in the scored parquets,
   checked directly, and no `muon:` block exists in the `ctag2d` workflow
   yamls' `event_weights` at all. **Update 2026-08-31/09-01**: ran the
   `jecshifts` workflow at scale for the first time (condor, MC-only, all
   4 eras, 80/84 datasets — 4 known-gone `bbH_Hto2Zto4L` failures; one job,
   `2022postEE/TTH_Hto2Z`, held on cgroup OOM at the default 3GB limit,
   fixed by hand-patching `request_memory=8000` into its `.sub` file since
   neither `submit_condor.py` nor `condor/submit.sub` expose a memory
   override). **Output actually has 8 shift categories, not 4** — `base`
   (nominal) plus `CMS_scale_j_<year>{Up,Down}`, `CMS_res_j_<year>{Up,Down}`,
   `CMS_scale_m_<year>{Up,Down}`, `CMS_res_m_<year>{Up,Down}` sibling
   directories per dataset — the workflow bundles JES/JER *and* muon
   scale/resolution together. MVA rescoring will need 9 passes (base + 8)
   per era, not the ~5 originally scoped. Completeness/resubmit check and
   the datacard-side consumption code are still ahead — see `second-brain/
   Tasks/HcZZ-fake-rate.md` "Update 2026-08-31/09-01 (overnight)".
   **CLOSED 2026-09-08**: rescoring (`_jecshifts_scored_v2`, dedup-bug-fixed
   per memory `hczz_jesjer_rescoring_dedup_fix`) and the actual datacard-side
   consumption code (`--jecshifts-dir` in `create_datacards_ctag2d_100150.py`)
   are both done — `CMS_scale_j`/`CMS_res_j`/`CMS_scale_m`/`CMS_res_m` are
   live shape systematics in the production datacard (r=412.0/kappa_c=74.11,
   see the "CURRENT" callout in the `ctag2d` variant section above). Muon
   ID/isolation/reco (efficiency SF, still distinct from the scale/resolution
   shift covered here) remains the one genuinely open piece of this item.
5. **Per-event HF-composition-style weight** — `hww-analysis`'s
   `higgs_hf.py` replaces a mis-scoped flat lnN on a pooled background group
   with a per-event GEN-jet-flavour weight. Same shape of problem as our
   `Other_Higgs`/`HPlusBottom` pooling (see the `lhe_pdf`/`scalevar`
   exclusion above). **Update 2026-08-19**: ported (`analysis/corrections/
   higgs_hf.py`, `higgsHFWeight: true` in both `ctag2d` workflow yamls),
   wired into `weight_manager()`, and smoke-tested — see the dated update in
   the Systematics section above. **Not yet wired into `create_datacards_
   ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts** — the column exists in
   the parquet, the datacard doesn't use it yet.
6. **`BR_HZZ4l`/`QCDscale_qqZZ` lnN rows** — **done 2026-08-17**, see the
   `ctag2d` variant subsection under Systematics above.

**Deliberately not adopted**: the multi-channel argmax MVA architecture
(SR + per-background CRs, floating a background via `rateParam` the way
`hww-analysis` floats `tt`). Decision 2026-08-14: keep the single
`mva_score_Signal` discriminant — the current 4-class MVA and all reference
r/κc numbers stay valid; this is a documentation and systematics-parity
effort, not a redesign.

---

## Update 2026-09-09: discriminant comparison against current production; `cjets` selection WP investigated and left unchanged

**Plain vs kappa vs 3-ratio, current ctag2d production**: reran the
plain/kappa-weighted/3-ratio discriminant comparison first done 2026-08-07/09
on the old pre-ctag2d pipeline — never checked against the current one before.
Result: on the current production (`_scored_v8`, `[100,150]`, no-ZX,
no-JES/JER scope), all three are within ~0.6% of each other at both stat-only
and full-syst tiers — no discriminant wins or loses meaningfully, unlike the
old pipeline's clear 3-ratio win (4-10%) and kappa's catastrophic full-syst
loss (6-25%), neither of which replicates here. Sanity-checked the plain-score
number bit-exact against the documented r=411.0 reference before trusting
anything. A first pass found 3-ratio apparently losing by 7.2% — turned out to
be a bin-count confound plus silent ROOT write-time corruption on a shared/
loaded host, not a real effect; worth remembering for any future
differently-binned comparison. Full writeup: second-brain
`Notes/ctag2d-3ratio-kappa-comparison-note.md`, memory
`hczz_ctag2d_3ratio_kappa_comparison`. New script (kept separate from
production, not wired in): `combine/scripts/create_datacards_ctag2d_compare.py`.

**Plain, WITH JES/JER, closing the scope gap above**: reran the production
script itself (`create_datacards_ctag2d_100150.py`, `--jecshifts-dir
.../hplusc_mva_4class_ctag2d_jecshifts_scored_v2`, no `--merge-bin-ranges`
so it lands on the same matched 20 bins as the comparison above) rather than
extending the compare script — only needs the one `mva_score_Signal` column,
same memory footprint as the JES/JER run that already succeeded 2026-09-08,
so no OOM risk this time. Both numbers reproduce the documented reference
bit-exact despite the different bin count: stat-only r=271.75/κc=49.79
(identical to the no-JES/JER stat-only number above, as expected — freezing
all constrained nuisances removes JES/JER's contribution along with
everything else), full-syst **r=412.0000/κc=74.11**, exact match to the
`_v8`/15-bin production reference. Kappa and 3-ratio with JES/JER are not
done yet — next step when picked back up.

**`cjets` object-selection working point — investigated switching to 2D,
decided against it**: the ctag2d migration only ever moved the c-tag SF
*weighting* to the 2D pseudo-continuous scheme — `cjets` object selection
(`atleast_one_cjet`) has stayed on the fixed 1D BTV Loose WP throughout.
Considered moving selection to 2D too (using category L0 as the "not-tagged"
boundary, mirroring "Loose"). Two independent reasons not to: (1) empirically,
on real `_scored_v8` qqZZ jets, "not-L0" passes only 35.6% of jets vs the
current 1D Loose WP's 94.4% — L0's boundary is a much tighter cut in practice
than "Loose" despite the naming parallel; (2) `Chirayu18/higgscharm`
(`hww-analysis` branch — the repo `ctag2d.py`'s SF-weighting code was ported
from) doesn't select on the 2D category grid anywhere either: their `cjets`
selection is also a fixed 1D WP (Medium, not Loose), and `CTag2DCorrector` is
applied only to `candidate_cjet` as a reweighting. Cross-checked the L0
boundary values themselves against the public HiggsDNA metaconditions
`Era2022_v1.json` "HPC_ctag_WPs" block — byte-for-byte match to what's already
in `ctag2d.py`, so the boundaries are right, it's the proposed selection *use*
of them that doesn't hold up. Code built and verified but deliberately not
wired into any production path: `get_ctag_mask_2d()`/`jet_ctagging_2d()`
(`analysis/working_points/utils.py`/`working_points.py`), side-by-side
workflow `hplusc_mva_4class_ctag2d_2dsel.yaml`. Full writeup: second-brain
memory `hczz_ctag2d_selection_wp_investigation`.

**Also cleaned up**: removed the stale "two checkouts not synced by git"
AFS/EOS-drift warnings and the "CURRENT" `ctag2d` reference above (was
r=413.0/2026-09-02, now r=412.0/2026-09-08 with JES/JER — see that callout
for the full supersession chain).

---

## Update 2026-09-11: true 4-era Z+X now included in the current (`_v8` + JES/JER) production

**Z+X (reducible background) inputs now cover all 4 Run 3 eras**, closing the
gap that had left every with-ZX number on record as a 2022-only (or older,
pre-muon-SF/pre-ctag2d) floor. The full chain, end to end:

1. **Raw Z+X control-region events** (fake-rate-weighted 3P1F/2P2F, all 4 eras
   — 2022preEE, 2022postEE, 2023preBPix, 2023postBPix) come from the standard
   `estimate_zx_background.py`-style output format, same schema
   `run_zx_inference.py` already expects.
2. **MVA-scored with this analysis's own model**: ran
   `scripts/run_zx_inference.py` over all 4 eras with the documented,
   checkpoint/config-verified pair (`models/best_model.pt` +
   `models/hc_zzto4l_mw_training_4class_nomass.yml`, MD5- and
   `global_features`-confirmed against the canonical config — see "Use
   `models/best_model.pt`..." in Known traps) to produce
   `mva_signal_score` per event for every era, output under
   `zx_background_mva_pkatris/{era}/zx_{3p1f,2p2f}_{era}_mva.parquet`.
3. **Datacard build** with the current signal-region production
   (`create_datacards_ctag2d_100150.py --scored-dir
   hplusc_mva_4class_ctag2d_scored_v8 --jecshifts-dir
   hplusc_mva_4class_ctag2d_jecshifts_scored_v2 --zx-dir
   zx_background_mva_pkatris --merge-bin-ranges "2-4,7-10"`) — i.e. the true
   4-era Z+X alongside the full `_v8` + JES/JER signal-region chain, not a
   partial-era substitute. Z+X yield in `[100,150]`~GeV: **133.24** (43% of
   an inflated bkg total once ZX is counted; roughly 2× the yield from an
   earlier 2-era-only pass, consistent with adding two more eras of real
   background rather than a small correction).
4. **`combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001
   --rRelAcc 0.00001`** on the resulting datacard.

**Result: r=480.0 (median), κc=85.89** — full quantile band 2.5%: r=219.4/
κc=40.70; 16%: r=311.0/κc=56.60; 50%: r=480.0/κc=85.89; 84%: r=788.0/
κc=139.26; 97.5%: r=1283.8/κc=225.15. This is looser than the no-ZX headline
(r=412.0/κc=74.11) and the earlier 2-era-only with-ZX floor (r=460.0/
κc=82.43), as expected — adding real background loosens the limit, it
doesn't tighten it. This supersedes every earlier with-ZX number on record
(r=368.0/329.0 at `[90,160]`, both pre-muon-SF and pre-every-ctag2d-fix; the
2-era-only r=460.0 floor above) as the current with-ZX reference for
`[100,150]`.

Output: `combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris/`
(`datacard_mva_with_zx.txt`, `histograms_mva_with_zx.root`,
`combine_result.log`). Full reproducible step-by-step (exact commands for
every stage above): `second-brain/Notes/hczz-zx-pkatris-4era-verification-2026-09-10.md`.

**Impact ranking, same datacard** (`b-hive_ttcc/combine/impact_ranking.py --real
--expect-signal 480.0 --r-range 0,1000 --parallel 4`, the real
`combineTool.py -M Impacts` profile-likelihood workflow, §7 above): 24
nuisances total (the 22 from the no-ZX headline plus `ZX_norm` and the
free-floating `ZX_rate` rateParam). **`CMS_ctag2d` still dominates by a wide
margin** (impact 252.66, ~7× the #2 nuisance) — even larger in absolute terms
than the no-ZX headline's 180.46, consistent with `CMS_ctag2d` reweighting
every MC background process that Z+X now sits alongside. **`ZX_rate` itself
ranks #2** (impact 34.23, post-fit 1.00 +0.33/−0.25) — expected, since it is
the one parameter directly controlling the newly-added background's
normalization. `CMS_scale_j` (JES/JER) ranks #3 (26.55), ahead of every lnN
row. All pulls ≈0 with post-fit uncertainty ≈ prefit, as expected for a blind
Asimov fit. Output: `impacts_real.json`, `impacts_real.pdf` in the same
directory. Full ranked table and the plot are in
`docs/hczz_ctag2d_jesjer_results.pdf` (pages 5–6, appended alongside the
yields/limits page for this result).

**Stat-only comparison, with vs. without Z+X** (`combine -M AsymptoticLimits
--freezeParameters allConstrainedNuisances` on both cards; `ZX_rate` still
floats since it's unconstrained, not frozen by this flag — re-verified
2026-09-11, both cards rerun fresh rather than trusting the earlier
documented no-ZX number):

| Quantile | r (no ZX) | κc (no ZX) | r (with ZX) | κc (with ZX) | Δr | Δr % |
|---|---|---|---|---|---|---|
| 2.5% | 140.12 | 26.92 | 144.30 | 27.65 | +4.18 | +3.0% |
| 16% | 189.48 | 35.51 | 197.02 | 36.82 | +7.53 | +4.0% |
| **50% (median)** | **271.75** | **49.79** | **282.00** | **51.57** | **+10.25** | **+3.8%** |
| 84% | 394.69 | 71.11 | 410.14 | 73.79 | +15.45 | +3.9% |
| 97.5% | 550.66 | 98.14 | 573.54 | 102.10 | +22.88 | +4.2% |

**Stat-only, Z+X only loosens the median limit by ~3.8%** — much smaller than
the ~16.5% full-syst loosening (r=412.0→480.0). Most of the with-ZX effect is
therefore driven by systematics, not the added statistical background
itself: the `ZX_norm` 30% lnN and the free-floating `ZX_rate` rateParam
(impact 34.23, ranked #2 in the full-syst impact table above) account for
the bulk of the gap between the stat-only and full-syst Z+X effect. Output:
`combine_result_statonly.log` in both output directories (`_with_zx_pkatris`
and the no-ZX `_v8_jecshifts` dir). Table also on page 7 of
`docs/hczz_ctag2d_jesjer_results.pdf`.

## Update 2026-09-11: PDF/scale systematics audit against CMS's own recipe
deck — 2 real fixes applied + reverified, 1 anomaly resolved as benign, 1
open question flagged (not fixed)

Triggered by reading a CMS internal talk on PDF/PS/scale-uncertainty
recipes (Markus Seidel, "CMS recipes for higher order, PS, and PDF
uncertainties" — `second-brain/slides/physicsdays_systematics.pdf`) and
checking this analysis's actual systematics code against it point by point.

**Fix 1 — `lhe_alphaS` re-included for `qqZZ`/`Other_Higgs`.** The
abs()-before-symmetrizing bug that forced this nuisance's Up to always raise
and Down to always lower every process's yield (documented in the
"Update 2026-08-19" section above, excluded everywhere as a result) was
actually fixed in `analysis/corrections/lhepdf.py` on **2026-08-28**
(commit `167ebca`) — 9 days after the exclusion was written — but
`create_datacards_ctag2d_100150.py`'s `USABLE_SYST` exclusion was never
revisited. Re-verified directly against the current v8 production (scored
2026-09-08, i.e. built from the already-fixed `lhepdf.py`):
`weight_lhe_alphaSUp >= 1` for only 57% of qqZZ events in `[100,150]` (not
the bug's forced 100%), confirming the sign is genuinely mixed again. Every
`Other_Higgs` sub-sample spot-checked sane in the same window (official
samples ±0.9-2.7%, `HPlusBottom` ±7.7%, consistent with the pre-bug
2026-08-04 characterization already on record). `ggZZ`/`Signal` stay
excluded — see the anomaly-resolution paragraph below. Commit `31b3df3` on
`HcZZ-zx-estimation`.

**Fix 2 — qqZZ `scalevar_muR`/`scalevar_muF` decoupled from `QCDscale_qqZZ`.**
Measured the actual normalization shift these two shape systematics imply
for qqZZ in `[100,150]`: `scalevar_muR` +1.9%/−1.5%, `scalevar_muF`
+2.8%/−3.9%, combining in quadrature to ~+3.4%/−4.2% — almost exactly on
top of the separate flat `QCDscale_qqZZ` lnN (4%). Traced the cause: the
shape templates were built as full replacement weights (norm+acceptance+
shape together, never decoupled), so the same ~4% normalization uncertainty
was being counted twice — once via the lnN, once baked into the shape.
This is the same double-counting the codebase already checks for and avoids
on the PDF side (`pdf_gg`/`lhe_pdf`, see the "no gap here" note above `qqZZ`
in `USABLE_SYST`) but had never been checked for the QCD-scale side. Fixed
by rescaling `scalevar_muR`/`muF`'s Up/Down histograms to the nominal
integral before writing them out (`NORM_DECOUPLED` set in
`build_histograms`), matching the CMS-recommended 3-way LHE-scale
decoupling (`physicsdays_systematics.pdf` slide 5 / HIG XS handbook
§I.4.2: normalization / acceptance / shape, drop normalization from the
shape template when a separate rate uncertainty already covers it). Commit
`9cd16b0`.

**Anomaly resolved as benign — ggZZ's `lhe_pdf`/`lhe_alphaS`/
`lhe_pdf_alphaS` degeneracy.** While re-verifying `lhe_alphaS`, found that
`weight_lhe_pdfUp`, `weight_lhe_alphaSUp` and `weight_lhe_pdf_alphaSUp` are
all *identical* for every ggZZ event — initially looked like a
column-aliasing bug (didn't match any branch in `lhepdf.py` on first read).
Traced it: all three are also identical to `weight_nominal` itself, because
each modifier's own per-event contribution is exactly `1.0` — genuinely no
PDF/αs variation for ggZZ (consistent with `lhepdf.py`'s no-real-variation
fallback, and with the existing "ggZZ has no LHEScaleWeight branch at all
and degenerate lhe_pdf/lhe_alphaS" note above). Ruled out a broader
aliasing bug by checking `weight_CMS_pileupUp`/`weight_ps_isrUp` on the same
events — those differ from nominal normally. No code change; `ggZZ`'s
narrow allow-list (already excluding `lhe_pdf`/`lhe_alphaS`) was already
correct. Commit `982f857`.

**Open question, flagged not fixed — `lhe_pdf`'s Hessian-vs-MC-replica
formula.** `lhepdf.py`'s own docstring cites LHA IDs `306000`–`306102` for
the `LHEPdfWeight` branch. Verified directly against the authoritative
LHAPDF `.info` files (`lhapdfsets.web.cern.ch`):

| LHA ID | Set | NumMembers | ErrorType |
|---|---|---|---|
| `306000` | `NNPDF31_nnlo_hessian_pdfas` — cited by `lhepdf.py` | 103 | `symmhessian+as` |
| `325300` | `NNPDF31_nnlo_as_0118_mc_hessian_pdfas` — CMS's documented default (`physicsdays_systematics.pdf` slide 4) | 103 | `symmhessian+as` |
| `303600` | `NNPDF31_nnlo_as_0118` — the genuine MC-replica base set | 101 | `replicas` |

Both sets actually plausible here are **Symmetric Hessian**
(`ErrorType: symmhessian+as`), not MC-replica — the "`_mc_`" in `325300`'s
name refers to how it was *derived* (Hessian-reduced from an MC-replica
fit), not its `ErrorType`. Per the CMS recipe deck's slide 3, symmetric
Hessian combination is `ΔX = sqrt(Σ(Xi − Xnom)²)` — **no division**.
`lhepdf.py`'s `delta_pdf = sqrt(Σ(diffs)² / 99)` divides by `N−1=99`, the
MC-replica statistical-variance formula (correct only for a set like
`303600`, `ErrorType: replicas`, which — being 101 members, not 103 — can't
be what's actually hitting the `Case 2 (103 members)` branch every process
here uses). This suggests the "2026-08-03" fix (which *added* the `/99`
division, framed as correcting an MC-replica-vs-Hessian mismatch — see the
2026-08-03 entry above) may have applied the wrong correction.

**Not fixed, deliberately**: reverting the `/99` division would inflate
every currently-used `lhe_pdf` value by `√99 ≈ 9.95×` — qqZZ's
currently-measured ~2.8% (already cited elsewhere in this doc as consistent
with HIG-24-013's official ~3.1-3.4% PDF(qq→ZZ) number) would balloon to an
implausible ~28%, contradicting the very official reference this doc uses
to justify not adding a redundant `pdf_qq` lnN. That tension means either
the eigenvector-vs-replica reasoning above is missing a convention specific
to how this analysis's `LHEPdfWeight` branch is laid out, or something else
is compensating — not resolved. Testing the corrected formula properly
needs the raw ~100-member `LHEPdfWeight` array from NanoAOD directly (not
retained in the scored parquets, which only keep the already-collapsed
`weight_lhe_pdfUp/Down` summary columns), so this is a reprocessing-level
investigation, not a script fix. **Left as-is pending that investigation.**

**Validation — reran the datacard build + combine with fixes 1 and 2
applied**, same v8+JES/JER+4-era-ZX configuration as the "Update 2026-09-11:
true 4-era Z+X" section above:

| Quantile | r (pre-fix) | r (post-fix) | κc (post-fix) |
|---|---|---|---|
| 2.5% | 219.3750 | 219.3750 | 40.70 |
| 16% | 311.0010 | 311.0010 | 56.60 |
| **50% (median)** | **480.0000** | **480.0000** | **85.89** |
| 84% | 788.0414 | 787.0847 | 139.10 |
| 97.5% | 1283.8478 | 1283.5974 | 225.11 |

**Median unchanged to 4 decimal places; 84%/97.5% shift by <0.15%.**
`CMS_ctag2d` dominates the total uncertainty by ~10× over the next-ranked
nuisance, so correcting these two subdominant nuisances doesn't move the
headline — a useful confirmation that r=480.0/κc=85.89 is robust to both
issues, not that the fixes were pointless. Output:
`combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris_systfix/`
(`datacard_mva_with_zx.txt`, `histograms_mva_with_zx.root`,
`combine_result.log`). Commits: `31b3df3`, `9cd16b0`, `982f857` on
`HcZZ-zx-estimation` (no code change for the third — comment/verification
only). Full session narrative:
`second-brain/Tasks/HcZZ-fake-rate.md` (2026-09-11 continued entry).

---

## Update 2026-09-13: full framework-vs-slide-deck audit — ggZZ central-value
k-factor was silently missing (~2.27×), now fixed and validated

Followed up the 2026-09-11 PDF/scale audit above with a from-scratch,
slide-by-slide consistency check of the entire framework against
`physicsdays_systematics.pdf` (all 16 slides, not just PDF/scale), not just
the two nuisances already touched. Confirmed correct: αs-member handling
(101/102 split cleanly from the 100 eigenvector members), ME/LHE
anti-correlated scale-corner discarding (`lhescale.py`), inclusive
cross-section normalization sourcing, and — notably, given this project's
history of up/down-swap bugs — the `ps_isr`/`ps_fsr` PSWeight index mapping
(`PSWeight[0]/[2]` and `[1]/[3]`, matches slide 12 exactly, no swap here).
Two gaps found:

**Hadronization / UE-color-reconnection (slides 14-15)**: no CR1/CR2/ERD
anywhere in the codebase — judged a legitimate omission for a 4ℓ +
jet-multiplicity analysis (CR/ERD matters most for top-mass-like or
substructure-sensitive measurements). b/c-fragmentation coverage is murkier:
`CMS_ctag2d`'s correctionlib JSON only exposes a combined `up_Total`/
`down_Total` (no per-source breakout used anywhere in the code, consistent
with the "13 sources missing" finding on record), and while the reference
charm-tag calibration paper (arXiv:2111.03027) lists c-fragmentation as a
distinct systematic source for this measurement type, nothing in the JSON is
explicitly labeled as such — can't resolve whether it's folded into `Total`
or simply absent without asking BTV/the calibration owners. Accepted as-is;
not worth chasing further given `CMS_ctag2d` already dominates the fit by
~10× and this sub-component wouldn't move the ranking either way.

**ggZZ k-factor (slides 9-10, "higher-order reweighting") — real bug,
fixed.** `create_datacards_ctag2d_100150.py` already carried a
`kfactor_ggZZ` = 1.10 `lnN`, correctly cited from HIG-24-013 Table 15
("gg→ZZ k-factor 10%") — but that 10% is the **uncertainty on** CMS's actual
gg→ZZ NNLO/LO k-factor, not the k-factor itself. The real central-value
correction (HNNLO-computed, standard CMS-PAS-HIG-19-001-style H→ZZ→4ℓ
convention) is **≈2.27 at m(ZZ)=125 GeV** — and it was never applied
anywhere: the ggZZ sample (`GluGluToContinto2Z*_mcfm701-pythia8`) is the
standard **LO** loop-induced MCFM701 sample, and an exhaustive grep for
`kfactor`/`k_factor`/`weight_kfactor`/`ggzz_weight` across `analysis/`
turned up zero hits outside that one `lnN` line. The analysis had been
silently under-normalizing its ggZZ background by a factor of ~2.27 for its
entire history.

**Fix** (`create_datacards_ctag2d_100150.py`, near `PROCESS_XS_PB`):

```python
GGZZ_KFACTOR = 2.27
EXPECTED_YIELDS = {p: xs * TOTAL_LUMI_PB for p, xs in PROCESS_XS_PB.items()}
EXPECTED_YIELDS["ggZZ"] *= GGZZ_KFACTOR
```

Applied once, to the shared `EXPECTED_YIELDS` dict that both
`load_mc_scored_parquets` (nominal + shape systematics) and
`load_jecshifts_scored_parquets` (JES/JER) read from — so it propagates
consistently through every ggZZ histogram, nominal and every Up/Down
variation alike, not just the final yield. The existing `kfactor_ggZZ` 1.10
`lnN` is unchanged — it now correctly sits as the uncertainty *on top of*
the corrected central value instead of being the only place the k-factor
appeared at all. The three sibling scripts that carry the same pre-fix
pattern (`create_datacards_ctag2d.py`, `_3ratio.py`, `_compare.py`) were
deliberately left untouched — only `_100150.py` is the live production
script for the current headline.

**Validation — reran the datacard build + combine**, same
v8+JES/JER+4-era-ZX configuration:

| Process | Yield in [100,150] (pre-fix) | Yield in [100,150] (post-fix) |
|---|---|---|
| ggZZ | 2.9418 | **6.6779** (= 2.9418 × 2.27, exact) |
| Total background | 227.58 | 231.31 |

| Quantile | r (pre-fix) | r (post-fix) | κc (post-fix) |
|---|---|---|---|
| 2.5% | 219.3750 | 219.3750 | 40.70 |
| 16% | 311.0010 | 311.0010 | 56.60 |
| **50% (median)** | **480.0000** | **480.0000** | **85.89** |
| 84% | 787.0847 | 786.1281 | 138.93 |
| 97.5% | 1283.5974 | 1280.7163 | 224.61 |

**Median unchanged to 4 decimal places; 84%/97.5% shift by <0.3%** — ggZZ is
a small enough background component (2.9→6.7 events against ~227 total)
that even a real, previously-missing factor-of-2.27 correction barely moves
the fit, for the same reason as the 2026-09-11 fixes: `CMS_ctag2d`
dominates the impact ranking by ~10×. Low headline significance, but a
genuine correctness fix — the physical ggZZ yield used throughout the
analysis was wrong by more than a factor of 2 until now. Output:
`combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris_ggzzkfactor/`
(`datacard_mva_with_zx.txt`, `histograms_mva_with_zx.root`,
`combine_result.log`). Committed (`06e832b`, propagated to sibling scripts
in `84b2282`) and pushed to `origin/HcZZ-zx-estimation`. Memory:
`hczz_physicsdays_systematics_audit.md`.

---

## Update 2026-09-14: `ZX_norm`/`QCDscale_qqZZ` provenance chased down —
`QCDscale_qqZZ` confirmed exact, `ZX_norm` cited and cross-checked against
pkatris's real production data (no code value change)

Same slide-audit follow-up as the ggZZ k-factor fix above, applied to the
two remaining lnNs that carried no (or a weaker) citation: `ZX_norm` (30%)
had **zero** citation anywhere in git history — every other lnN in the same
block cites either HIG-24-013 Table 15 or the Felix Heyen thesis. A
competing "40%" figure was found in `second-brain/Tasks/CMS-AN.md`, but that
doc turned out to be an unreliable, never-updated generic checklist (its own
"ggZZ cross-section 10%" entry has the *same* mislabeling the real ggZZ bug
had — the 10% uncertainty value mistaken for the correction itself — which
undermines it as an independent cross-check).

**Found the real HIG-24-013 text — it's arXiv:2501.14849** ("Measurements of
Higgs boson production cross section in the four-lepton final state...
13.6 TeV"). Its systematics section:

> "The following systematic effects in the reducible ZX background
> determination are studied: the statistical uncertainty in the number of
> events in the control regions, the effect of ±1σ variations of the
> misidentification rates, and the difference in composition among various
> processes that contribute to this background. The overall effect of these
> three sources ranges between 25% and 46%, depending on the final state,
> and is included as a nuisance parameter with a log-normal prior in the
> fit."

There is no single official number — it's per final state (4e/4μ/2e2μ/2μ2e),
25-46%. This card uses one unsplit "hczz" bin (no per-final-state channels),
so a flat number is a pre-existing structural simplification, not something
this session introduced. **30% sits inside the real range.**

**Same paragraph area also confirms `QCDscale_qqZZ`=4% exactly** — the paper
states "this yields an overall 4% effect" for the qq→ZZ scale variation.
High confidence, not a placeholder; cited in code now.

**Independent cross-check of the CR-statistics component, using the actual
production data** (not the stale pre-pkatris CR sample an earlier check
mistakenly used — see the note below): read `zx_weight`/`zx_weight_sq`
directly from `zx_background_mva_pkatris` (the same parquets
`load_zx_parquets` reads for the real combine run) across all 4 eras.
`sqrt(Σ zx_weight_sq) / Σ zx_weight` = **5.61%** (total yield 133.2386,
matching the production's own output exactly — confirms this used the right
dataset). Per era: 2022preEE 11.3%, 2022postEE 8.0%, 2023preBPix 11.0%,
2023postBPix 18.3% (smallest-yield era, largest relative stat error). The
per-event `zx_weight_up` systematic variation column shifts the total by
−2.68%. Neither of these, even combined, approaches 25-46% — consistent
with the real HIG-24-013 uncertainty being dominated by the misID-rate/
composition systematics rather than raw CR statistics for a large real
dataset, not a contradiction.

**Correction to an earlier same-day cross-check**: a first pass at this used
`estimate_zx_background.py`'s default data source
(`hplusc_mva_4class_CR`/`hplusc_mva_4class_CR_merged`) and got 22.7%
(2022preEE only) — this is **this project's own separate, pre-pkatris CR
production**, not pkatris's data, so it wasn't actually validating the
number the headline result uses. Redone correctly above.

**Verdict: `ZX_norm`=30% is accepted as a defensible representative point in
the real 25-46% range** — not precisely derived (the misID-variation and
composition-difference components aren't independently quantified in this
codebase), but not a placeholder needing correction either, unlike the ggZZ
k-factor. **No numeric code change** — citations added to
`create_datacards_ctag2d_100150.py` and `create_datacards_ctag2d.py` (the
two scripts carrying `ZX_norm`) documenting this investigation inline.

**Unrelated bug surfaced while chasing this, flagged separately, not
investigated further**: `estimate_zx_background.py`'s
`estimate_zx_ss_crosscheck()` function (added 2026-08-28, apparently never
run before) gives Net Z+X(SS) = 2938 for 2022preEE — ~500× the OS method's
5.94, clearly broken (likely the same `sumw=0.0` xs-lookup failure the OS
method already warns about for this era, or a CR-selection mismatch). Open
item for a future session.

**`BR_HZZ4l`=2% chased the same day, inconclusive-but-plausible verdict
(not a clean confirm like `QCDscale_qqZZ`, not a bug).** Downloaded and
read the actual HIG-24-013 PDF (arXiv:2501.14849) directly — its systematics
section (Section 8) states outright: *"The effect of theoretical
uncertainties on the signal is negligible for the present analysis and is
not included in the fit."* The paper this codebase cites for every other
lnN doesn't apply a signal branching-ratio uncertainty at all; there's no
number in it to match `BR_HZZ4l` against.

Checked the primary source instead — the paper's own reference [84], LHCHWG
"Standard Model Higgs-Boson Branching Ratios with Uncertainties"
(arXiv:1107.5909): *"Both TU and PU on the important channels H→ZZ and
H→WW remain at the level of 1% over the full mass range, giving rise to a
total uncertainty below 3%."* That's a **1-3% range, not a single quoted
point value** — 2% sits inside it (plausible) but isn't independently
pinned down the way `QCDscale_qqZZ` was.

**Verdict**: `BR_HZZ4l`=2% is plausible and not contradicted, but genuinely
inconclusive rather than confirmed. HIG-24-013 not using this uncertainty
at all doesn't make 2% wrong for *this* analysis — a κc coupling-extraction
search is more sensitive to the assumed B(H→ZZ→4l) than a fiducial
cross-section measurement is, so including it (per the Felix Heyen thesis,
this lnN's original source) is a reasonable, independent methodological
choice that HIG-24-013 simply doesn't need to make for its own measurement.
**No numeric code change** — inline citation notes added to all 3 scripts
carrying `BR_HZZ4l` (`create_datacards_ctag2d_100150.py`,
`create_datacards_ctag2d.py`, `create_datacards_ctag2d_compare.py`).

Memory: `hczz_physicsdays_systematics_audit.md`.

---

## Update 2026-09-14 (later): no-ZX headline rebuilt with every current fix
applied together for the first time — r=412.0→**411.0**, κc=74.11→**73.93**

Found while doing a correctness pass on `second-brain/slides/analysis_overview_hczz.tex`:
the no-ZX headline datacard (`combine_run3_100150_ctag2d_v8_jecshifts/datacard_no_zx.txt`)
had never actually been rebuilt with `lhe_alphaS` re-inclusion, the qqZZ
norm/shape decoupling fix, or the `kfactor_ggZZ` central-value fix — all
three (2026-09-11 through 09-14) were only ever validated against the
**with-ZX** card (`_with_zx_pkatris_systfix`/`_ggzzkfactor`). Confirmed by
grepping the actual datacard: 22 rows, no `lhe_alphaS` row.

**Rebuilt properly**: same command as the with-ZX build, minus `--zx-dir`
(`create_datacards_ctag2d_100150.py --scored-dir hplusc_mva_4class_ctag2d_scored_v8
--jecshifts-dir hplusc_mva_4class_ctag2d_jecshifts_scored_v2 --skip-zx
--merge-bin-ranges "2-4,7-10"`), into
`combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_ggzzkfactor_nozx/`.
This picks up all 3 fixes at once since they live in the shared script.

**Result**: `combine -M AsymptoticLimits` gives median **r=411.0000**
(2.5%=195.87, 16%=273.18, 84%=646.09, 97.5%=999.82), **κc=73.93** — was
412.0/74.11 with the pre-fix 22-nuisance card, confirming the previously
*assumed* <0.3% shift (−0.24% exactly) rather than just citing the with-ZX
cross-check as a proxy.

**Re-ranked with the real 3-stage `impact_ranking.py -M Impacts`**
(`b-hive_ttcc/combine/impact_ranking.py --real --expect-signal 411.0`, 23
nuisances):

| Rank | Nuisance | Impact on r |
|---|---|---|
| 1 | `CMS_ctag2d` | 177.16 |
| 2 | `ps_isr` | 20.53 |
| 3 | `QCDscale_qqZZ` | 16.90 |
| 4 | `BR_HZZ4l` | 12.71 |
| 5 | `CMS_scale_j` | 12.14 |
| 6 | `QCDscale_gg` | 10.72 |
| 7 | `CMS_eff_e_reco_below20` | 9.99 |
| 8 | `kfactor_ggZZ` | 9.18 |
| 9 | `higgs_plus_c` | 8.84 |
| 10 | `pdf_gg` | 8.81 |
| ... | | |
| 21 | `scalevar_muF` | 0.12 |

**Notable finding**: `scalevar_muF` collapsed from **#4 (impact 16.15) to
#21 (0.12)** — the 2026-09-11 norm/shape-decoupling fix had only ever been
demonstrated on the with-ZX card; this is the first time it actually took
effect on the no-ZX headline, and the effect is dramatic (the systematic's
shape-driving power was almost entirely the double-counted normalization
shift, now removed).

**Also re-ranked the with-ZX `_ggzzkfactor` card** (its existing
`impacts_real.json`/`.pdf` predated the same-day fixes, from the morning of
09-11 before `_systfix`/`_ggzzkfactor` were built): r=480.0 unchanged, but
the ranking moved — `CMS_ctag2d` 252.66→249.98, `ZX_rate` #2 34.23→35.39,
`CMS_scale_j` #3 26.55→26.40, same `scalevar_muF` collapse (#23, 0.10).

**Also reran stat-only** (`--freezeParameters allConstrainedNuisances`) on
both final rebuilt cards: no-ZX 273.5 (was 271.75), with-ZX 283.5 (was
282.00) — the kfactor fix changes ggZZ's central value, not just its
uncertainty, so even the frozen-nuisance fit shifts slightly. Z+X stat-only
effect is still small (+3.7%) vs. full-syst (+16.8%) — same
systematics-driven conclusion as before, just with refreshed numbers.

Output: `combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_ggzzkfactor_nozx/`
(datacard, histograms, `impacts_real.json`/`.pdf`, both AsymptoticLimits
logs). `second-brain/slides/analysis_overview_hczz.tex` updated throughout
(systematics counts 22/24→23/25, impacts table, "Where we stand" table,
mistakes-found trail, priorities table, HWW+c comparison) — not yet
committed. Memory: `hczz_ctag2d_l0_updown_swap_fix`,
`hczz_jesjer_datacard_complete`, `hczz_zx_pkatris_production_and_era_bug`.

---

## Update 2026-09-15: L0 Up/Down swap correction RETRACTED — official
ctag2d SFs used as-is; new headline **no-ZX r=321.0/κc=58.33, with-ZX
r=329.0/κc=59.72**

**Why**: the 2026-09-02 swap "fix" (`analysis/corrections/ctag2d.py`,
`_eval_flat_updown_corrected()`) was re-investigated at component level,
going beyond the original `Total` vs. `Stat` comparison. Checked every
individual source (`up/down_Stat`, `up/down_JES`, `up/down_PUWeight`) for
L0 across a 5-point pT grid, live against the real
`flavTaggingSF_2022preEE.json.gz`:

| Component | Pattern across pT |
|---|---|
| `Stat` | Consistently normal (up>down) at every point |
| `JES` | **Mixed** — inverted 4/5 points, normal at one |
| `PUWeight` | **Mixed** — inverted 2/5 points, normal at 3/5 |
| `Total` | Consistently inverted; `up_Total` pinned at exactly 0.3000 regardless of pT |

A genuine swap-then-combine bug should leave a uniform signature across
every component feeding `Total`; a genuine intentional anti-correlation
(L0 as a complement category) should too. Neither holds — `Stat` alone is
clean, `JES`/`PUWeight` are inconsistent with no pT pattern. Additionally,
the "fix" itself was only ever a magnitude-based relabeling of the file's
own two reported numbers (`up_corrected = max(up_flat, down_flat)`), never
a recomputation from components — too thin a basis to keep silently
altering an official calibration's reported uncertainty for two weeks
without confirmation.

**Action taken**:
1. **Reverted the correction** in `analysis/corrections/ctag2d.py`
   (`add_weights()` now calls `_eval_event(SYST_UP)`/`_eval_event(SYST_DN)`
   directly — the swap-detection function was removed entirely, not just
   disabled). Applied to **both** real checkouts confirmed non-identical by
   inode: `/eos/home-s/snandaku/Higgscharmnew/...` (same inode as
   `/eos/user/s/snandaku/Higgscharmnew/...`, i.e. one file) and the
   genuinely separate `/afs/cern.ch/user/s/snandaku/Higgscharmnew/...`
   checkout. A third copy, `/afs/cern.ch/user/s/snandaku/higgscharm/...`
   (no `Higgscharmnew`), was checked and never had the swap fix at all —
   nothing to revert there.
2. **Reported to the calibration's maintainer** (Livio, file owner) by
   email 2026-09-15 — full finding, evidence (`up_Stat`/`down_Stat`
   cross-check, bit-exact `up_Total`==`down_Stat` match), and an explicit
   note on what we did *not* claim (an earlier mechanism hypothesis about
   L0's Stat sources being unbinned in pT was checked and found false
   before sending). Not yet confirmed by BTV/POG either way.
3. **Rebuilt results using the official SFs, unmodified** — without a full
   condor reprocessing campaign. The existing `hplusc_mva_4class_ctag2d_scored_v8`
   parquets still carry the swap-corrected `weight_CMS_ctag2d_<year>Up/Down`
   baked in from when the correction was live, so a new script,
   `combine/scripts/create_datacards_ctag2d_100150_rawsf.py`, recomputes
   just that one weight column directly from the jet-level branches already
   stored in the same parquets (`jet_btagPNetCvL/CvB/hadronFlavour/pt`),
   using the exact 3-leading-jet cap and category logic `CTag2DCorrector`
   uses, but reading `up_Total`/`down_Total` raw. Verified against a real
   sample before the full run: single-jet-dominated events reproduce an
   exact swap-back of the stored corrected values; multi-jet events show
   the expected partial effect; unaffected events match exactly.

**New results** (both cards, same v8+JES/JER+kfactor_ggZZ+lhe_alphaS+
decoupling production, only the ctag2d treatment changed):

| | no-ZX (23 syst) | with-ZX (25 syst) |
|---|---|---|
| Full-syst median $r$ | **321.0000** (was 411.0) | **329.0000** (was 480.0) |
| $\kappa_c$ | **58.33** (was 73.93) | **59.72** (was 85.89) |
| Stat-only median | 273.5 (**unchanged**) | 283.5 (**unchanged**) |
| `CMS_ctag2d` impact | 147.74 (was 177.16), #1, ~4.1× #2 | 162.04 (was 249.98), #1, ~4.0× #2 |

Stat-only limits identical to the retracted-fix version confirms this is
100% a systematics-shape effect, not a yield change — same verification
pattern used when the swap fix was originally applied, now applied in
reverse.

**Real ranking changes worth flagging**:
- No-ZX: `QCDscale_qqZZ` fell from #3 (16.90) to #14 (3.45);
  `CMS_eff_e_reco_below20`, `CMS_scale_j`, `higgs_plus_c`, `BR_HZZ4l`,
  `lhe_alphaS` all moved up into the top 6 now that `CMS_ctag2d` isn't as
  overwhelmingly dominant.
- With-ZX: **`ZX_rate` collapses from #2 (35.39) to #13 (6.19)** — the
  single biggest ranking change in the whole revert. With `CMS_ctag2d` no
  longer artificially inflated, the fit doesn't need `ZX_rate` to absorb as
  much slack. `QCDscale_qqZZ` similarly falls to #14 (2.63).

**A real transient issue hit twice during this rebuild, for the record**:
`text2workspace.py`/`combineTool.py -M Impacts` failed deterministically
twice in a row reading the freshly-written EOS ROOT file (`Bogus norm 0.0`
for `CMS_eff_m_id Down`, `qqZZ`) — but `uproot` read the exact same
histogram correctly both times (sum=36.29, not zero), and copying the
datacard+ROOT file to local `/tmp` first let `text2workspace.py` read it
without error. Consistent with the documented EOS-read-race pattern
(README "Known traps") — not real data corruption. All combine/Impacts
runs for this update were done against local copies, then results copied
back to the EOS output directories for permanence.

**Not done, deliberately left open**: no condor resubmission, no NanoAOD
reprocessing — this revert only needed the one recomputed weight column,
not the full production. Still haven't separately flagged this to Chirayu
(HWW+c), who reads the identical file/correction and would be affected the
same way if the swap turns out to be real after all.

Output: `combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_ggzzkfactor_nozx_rawsf/`
and `combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris_ggzzkfactor_rawsf/`
(datacards, histograms, `impacts_real.json`/`.pdf`/`-1.png`, `workspace.root`,
all `higgsCombine*.root`, `combine_result.log`/`combine_result_statonly.log`).
New script: `combine/scripts/create_datacards_ctag2d_100150_rawsf.py`.
`second-brain/slides/analysis_overview_hczz.tex` updated throughout (the
"What the calibration costs" frame retitled and rewritten, both impacts
frames rebuilt with new plots/tables, uncertainty-band plot's black/red
roles swapped, "road so far"/"Where we stand"/"Mistakes found and
fixed"/Priorities/HWW+c-comparison all updated) — not yet committed.
Memory: `hczz_ctag2d_l0_updown_swap_fix` (superseded by this update).

---

### Update 2026-09-16: Z+c CR ctag2d 2D-SF validation check — small mixed
effect from the SF, but a much larger pre-existing Data/MC normalization
gap found in the CR itself (unrelated to ctag2d)

Independent check of whether the official 2D ctag SF improves data/MC
agreement in the `ztomumu`/`ztoee` `CR_Zplusc` control region (orthogonal
to the SR and the MVA score). No new condor production — reused the
existing 2022postEE `CR_Zplusc` parquets (the only era this CR was ever
submitted for).

**Scope caveat**: only the `cjets` (already c-tag-'loose'-passing) subset
has per-jet CvL/CvB/hadronFlavour/pt stored in these parquets, not the full
`jets` collection the SR's `CTag2DCorrector` uses (up to 3 leading jets of
any flavor). Applied the SF to the `cjets` subset only, per user direction
— a real but bounded scope reduction from the SR, not a full reproduction.

For each `cjets` entry, categorized into the same 11 WP codes (L0/C0-C4/
B0-B4) via the real `_category_np`, and histogrammed Data vs MC with and
without the central per-jet SF applied.

**Result**: the SF moves individual category ratios by roughly ±5-15%
(e.g. ztomumu C0: 0.543→0.507; C2: 0.500→0.546) but the total barely moves
(ztomumu 0.524→0.521, ztoee 0.594→0.591) — no systematic improvement or
degradation, shifts largely cancel.

**Bigger finding**: Data/MC ≈ 0.5-0.6 in every category, both channels —
MC over-predicts data by roughly a factor of ~1.7-2x. Confirmed via the
`base` category (no c-jet requirement at all): `ztoee` `base` gives
Data/MC = 0.616 and `ztomumu` `base` gives Data/MC = 0.556 (10,749,839
data / 19,329,379.6 MC) — same ballpark in both channels — **this deficit
predates and is unrelated to the c-tag requirement or the ctag2d SF**. Not
root-caused (out of scope) — candidates: missing/incomplete trigger
efficiency correction, a DY xsec/k-factor mismatch, or this production
simply never having had its absolute normalization validated before (built
for tagging-shape comparisons, not an absolute-yield CR). Flag for separate
investigation before drawing any absolute-normalization conclusion from
this CR.

Incidental finding from the `ztomumu` `base` rerun: 8 parquet files under
its output tree (`DYJetsToLL_50`, `TTtoLNu2Q`, `TWminusto2L2Nu`,
`TbarWplusto2L2Nu`) actually carry the `ztoee`-shaped schema
(`dielectron_mass`/... branches) — a small amount of `ztoee` output mixed
into `ztomumu`'s directory tree. Negligible in scale (8 of many thousands
of files) and doesn't move the 0.556 number; flagged, not fixed. Separately
4 files hit real Parquet corruption/truncation — matches the
already-documented EOS-read-race flakiness pattern, not new.

Script: `combine/scripts/create_cr_zplusc_ctag2d_check.py` (+
`check_zplusc_base_normalization.py` for the base-category cross-check).
Plots: `second-brain/slides/make_hczz_cr_zplusc_ctag2d_plots.py` ->
`hczz_cr_zplusc_{ztomumu,ztoee}_{nosf,withsf}.pdf`. Full writeup:
`second-brain/Notes/hczz-zplusc-cr-ctag2d-sf-check-2026-09-16.md`.

---

### Update 2026-09-16 (later): `hplusc_mva_4class_CR` feature-completeness +
isolation-double-veto fixes, pyarrow write bug root-caused/fixed, full
condor completion — this is the real path to a trustworthy with-Z+X number

**Part 1 — feature completeness.** Instrumented `run_zx_inference.py`'s
real feature-lookup functions against a real pkatris Z+X file: 33 of 101
MVA input features (`jet_phi`, most lepton charge/iso/ID variables)
silently defaulted to 0.0 — never listed in `hplusc_mva_4class_CR.yaml`'s
`histogram_config.axes` at all. Added the 66 missing axes to
`hplusc_mva_4class_CR.yaml` + `_loose_z2.yaml`, verified every field name
against `select_zzto4l_leptons()`'s `ak.zip` harmonization, verified
object/event selection byte-identical before/after (histogram config only).
Also fixed `run_zx_inference.py` to reconstruct per-jet arrays from
pkatris's `_0/_1/_2`-scalar file layout before scoring — his files store
only the leading jet, ours (via `estimate_zx_background.py`) already store
real per-jet arrays correctly; not our bug, but needed a shim to read his.

**Part 2 — real selection bug found.** Raw (non-FSR-corrected)
`pfRelIso03_all < 0.35` applied directly in muon `is_tight`, double-voting
against `select_zzto4l_leptons()`'s own FSR-corrected version applied
afterward — the exact mistake `zzto4l_CR.yaml` already carries a comment
warning against. Found in **both** `hplusc_mva_4class_CR.yaml` and, more
consequentially, `zplusl_os.yaml` (the fake-rate *measurement* workflow —
could have biased the measured fake rate itself, not just CR
classification). Fixed in both, both checkouts (AFS/EOS had drifted in
opposite directions on the two files — checked per-file, not assumed).

**Part 3 — the real production blocker: a deterministic pyarrow bug, not
flakiness.** `GluGluToContinto*`/`GluGlutoContinto*` (ggZZ) and `ZZto4L`
jobs had failed every single resubmission round for ~28 rounds with
`pyarrow.lib.ArrowNotImplementedError: Lists with non-zero length null
components are not supported`, raised in `dump_pa_table`'s
`ak.to_parquet` write — previously misattributed to generic flakiness.
Live-reproduced in the real container (`coffea-base-almalinux9:0.7.30-
py3.10`, pyarrow 19.0.1 / awkward 2.8.1): two specific awkward type shapes
can't be written — `N * option[var * ?T]` (a length<=1 "best candidate"
list, e.g. `m4l_3p1f`) and `N * option[var * T]` (genuinely multi-valued,
e.g. `cjets_h_dphi_3p1f`). Fixed in `analysis/utils/parquet_writer.py`
(`_fix_option_in_list()`): `ak.firsts()` for the first shape,
`ak.fill_none(field, [], axis=0)` for the second (preserves real
multi-jet lists — `ak.firsts()` there would have silently dropped jets
2/3, verified empirically before choosing the fix, not assumed).

**This session (continuation): confirmed the bug was far broader than
diagnosed, and the fix works.** Re-scanned all `hplusc_mva_4class_CR`
`.err` logs predating the fix (applied 2026-09-16 17:45 CEST): **1,027
error logs across the 4 eras** (200 / 326 / 275 / 226 by era) hit this
exact `ArrowNotImplementedError` — not just ggZZ/ZZto4L, also `DYJetsToLL`
and others. This is why 452 of the 589 expected `hplusc_mva_4class_CR`
job-chunks across all 4 eras still had no valid output going into this
session, despite the earlier 580-job full resubmission. Verified the fix
on the 12 `.err` logs written *after* 17:45 (the earlier session's targeted
ggZZ/ZZto4L resubmission): zero `ArrowNotImplementedError` remaining; the
only failures left are transient `XRootD Operation expired` /
`FileNotFoundError: No servers available` network errors — unrelated,
expected background noise. Refreshed the VOMS proxy (`voms-proxy-init
--voms cms`, `move_proxy()` in `submit_condor.py` copies it to
`/afs/.../private/`) and resubmitted all 452 missing jobs across all 4
eras via `jobs_status.py`'s built-in resubmit path (reuses the existing
`.sub` files verbatim, so the `workday` job-flavour from the original
submission is preserved — **not** `runner.py`, which silently defaults to
`longlunch` and has previously killed slow datasets). Confirmed via
`condor_q`: exactly 452 jobs queued, 0 held. Running as of this writing.

**Not committed** (`analysis/utils/parquet_writer.py`,
`hplusc_mva_4class_CR{,_loose_z2}.yaml`, `zplusl_os.yaml` fixes) — user
commits per standing preference.

**Correction, same session, after the 452 jobs finished**: checking
completeness with plain `jobs_status.py` (no `--eos`) afterward gave the
exact same "finished" counts as before the resubmission — a red flag.
`.err` logs show these jobs `xrdcp` output straight to
`root://eosuser.cern.ch//eos/user/s/snandaku/higgscharm/outputs/...`
(confirmed: 0 files modified in the plain AFS-local `analysis/outputs/`
tree in the prior 6 hours despite 452 jobs having just run) — **`--eos` is
the correct flag for this workflow's completeness check**, opposite of
the JES/JER campaign's "don't trust `--eos`" note (that was about which
*checkout* runs the code, a different question from where output lands —
always verify empirically per workflow, check a real `.err`'s
`xrdcp`/`cp` line). Re-checked with `--eos`: real completeness was
338/589 (not 137/589) — the 452-job resubmission was mostly redundant
(most already had valid EOS output the wrong check couldn't see) but not
harmful. Resubmitted the real 251 still-missing jobs the same way,
confirmed via `condor_q`: exactly 251 queued.

**Next**: once these 251 jobs complete (`jobs_status.py -w
hplusc_mva_4class_CR -y <era> --eos`, all 4 eras), merge, rerun
`estimate_zx_background.py` (ours, already verified correct) once
pkatris's `fake_rates.pkl` arrives, rerun the fixed `run_zx_inference.py`,
rebuild the with-ZX datacard, rerun combine. **Do not cite any with-ZX
number as final until this full chain reruns** — every prior with-ZX
number used either pkatris's jet-collapsed files or a stale/incomplete
July 27 production.

Full writeups: `second-brain/Notes/hczz-zx-feature-completeness-cr-fixes-
condor-2026-09-16.md`, `second-brain/Notes/hczz-zx-ctag2d-l0-flavors-and-
pyarrow-fix-2026-09-16.md`.

---

### Update 2026-09-16 (later still): qqZZ CR / On-Z CR ctag2d 2D-SF check,
full SR scope — SF has essentially no effect, agrees with the Z+c CR check

Second-priority task, done in parallel with the Z+X condor completion
above. Full-SR-scope counterpart to the Z+c CR check earlier the same day
(which was limited to the `cjets`-only subset that CR's parquets store).
Reused the already-complete `hplusc_mva_4class_ctag2d_scored` production's
`base` category (all 4 eras, same tree the real datacards use) — no new
condor production needed.

Regions: On-Z CR (`|m4l-91.1876|<7`), qqZZ CR (`m4l>180`), both m4l-only
selections orthogonal to the MVA-score fit discriminant. Wrinkle (opposite
of the Z+c CR case): this production's `weight_nominal` already has the
central ctag2d SF baked in, so "with SF" = `weight_nominal` as-is,
"without SF" = `weight_nominal` divided by a freshly recomputed per-event
product SF over up to 3 leading jets of **any** flavor — the real SR scope
(`events.selected_jets`), reusing `create_datacards_ctag2d_100150_rawsf.py`'s
`_raw_event_sf` per-jet recipe.

**Result**: On-Z CR Data/MC 1.078 (no SF) → 1.075 (with SF); qqZZ CR 1.098
→ 1.095. Per-process shifts from the SF: qqZZ/ggZZ/Other_Higgs all under
0.5%; Signal shows a large relative shift but ~0 absolute yield in these
windows (statistically meaningless). **The SF has essentially no effect in
either region** — expected, since qqZZ/ggZZ are light-jet-dominated with
low jet multiplicity/few real c-jets, and the SF's corrections concentrate
in c-tag-enriched WP cells. Per-m4l-bin breakdown (script log/JSON) shows
the same pattern: shifts under ~0.5% everywhere, data statistical
fluctuations (5-65 events/2-10 GeV bin) dominate any visible ratio
movement, not the SF.

**Combined with the Z+c CR result**: both CRs checked now agree — the
ctag2d SF neither clearly helps nor hurts Data/MC agreement in any CR
checked so far. Genuine, expected null result (CRs that aren't c-tag-
selected shouldn't feel much from a c-tag SF), not a check failure. Does
**not** reopen or contradict the 09-15 L0 swap retraction — that concerns
the up/down uncertainty *shape*'s SR impact, a different question from the
central SF's CR-normalization effect answered here.

Script: `combine/scripts/create_cr_qqzz_onz_ctag2d_check.py`. Plots:
`second-brain/slides/make_hczz_cr_qqzz_onz_ctag2d_plots.py` ->
`hczz_cr_qqzzonz_{onz,qqzzcr}_{nosf,withsf}.pdf`. Full writeup:
`second-brain/Notes/hczz-qqzz-onz-cr-ctag2d-sf-check-2026-09-16.md`.

---

### Update 2026-09-17: `hplusc_mva_4class_CR` output validated (not just
condor exit status) — clean, and completeness trend to 95%+

Per explicit ask: don't just trust that condor jobs leaving the queue
means the output is good. Validated directly:

- File counts sane at each checkpoint (e.g. 92/182/179/109 `.coffea`
  across the 4 eras), no zero-byte files, total sizes 412M-1.2G/era.
- `.coffea` files are lz4-compressed (`coffea.util` format), **not** raw
  pickle — an initial check with plain `pickle.load` falsely flagged
  32/32 sampled files as "corrupt" (`invalid load key '\x04'`); redone
  with `coffea.util.load`, all 32 loaded cleanly. Their content is
  metadata/cutflow only (sumw, run/lumi, per-CR-category counts) — the
  real physics payload lives in the parquet files, consistent with this
  workflow's already-documented "no coffea histograms for DY/WZ/tt"
  behavior (see the Z+X CR plots note above).
- Parquet content (what `estimate_zx_background.py`/`run_zx_inference.py`
  actually read downstream): 160/160 randomly sampled files across all 4
  eras read cleanly via `pyarrow.parquet.read_table`, all non-empty —
  including a targeted check of the exact datasets previously hit by the
  pyarrow bug (ggZZ, ZZto4L, DY, TT): all clean.
- Error-log sweep: 0 of 750 post-fix `.err` logs show
  `ArrowNotImplementedError`. Of the 367 that do have some traceback: 332
  `OSError` (XRootD timeouts), 9 `ValueError` ("not a ROOT file: first
  four bytes are b'\x00\x00\x00\x00'" — a corrupted/truncated transient
  read, same family), 4 `FileNotFoundError`, 4 `AttributeError`
  (misleading at first glance — traced to awkward's lazy-array wrapper
  surfacing the same underlying `XRootD Operation expired` when a branch
  like `pfRelIso03_all` is first accessed, not a real code bug). **No new
  bugs found** — the remaining failure tail is ordinary transient network
  flakiness, not a systematic issue.

**Completeness trend** across repeated `--eos`-checked resubmission
rounds: 137/589 (wrong flag, see the earlier correction above) →
338/589 → 551/589 → 562/589 (95.4%) → one more round in flight. Classic
diminishing-returns long tail from transient network reads, not a
systematic bug (confirmed by the error-log sweep above). Once this
converges (target: accept ≥97-98% rather than chasing every last file —
check what fraction of total statistics the remaining gap represents),
proceed with merge → `estimate_zx_background.py` → `run_zx_inference.py`
→ with-ZX datacard → combine, per the plan above.
