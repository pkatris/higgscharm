#!/usr/bin/env python3
"""
Rebuild docs/hczz_ctag2d_jesjer_results.pdf with the current [100,150] numbers (2026-09-23).

Changes since the 2026-09-21 version:
  - autoMCStats(10) on in both cards (production decision 2026-09-19); its per-bin
    prop_bin* nuisances now appear in the impact tables.
  - Z+X re-estimated from the hplusc_mva_4class_CR_merged production (pkatris's
    WZ-subtracted OS fake rates): 32.60 events (was 31.11 from the old CR production).
    Jet-inclusive, no c-jet requirement (user decision 2026-09-23).
  - Free ZX_rate rateParam removed: no Z+X control region in the fit, so it was
    constrained only by the SR Asimov and double-counted ZX_norm. ZX_norm (30% lnN)
    alone carries the Z+X normalization, as in HIG-24-013.
  - Stat-only limits recomputed on the current cards.
  - New page: CMS_ctag2d -> ZX_norm impact test.
Other_Higgs lhe_pdf/scalevar coverage (09-20), JES/JER, official ctag2d SFs (L0 swap
correction reverted since 09-15): unchanged from the 09-21 version.

Builds table/text pages with matplotlib, merges in the real combineTool.py -M Impacts
plot PDFs (made with combine/scripts/plotImpacts_privatework.py) via pypdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
from pypdf import PdfReader, PdfWriter

OUTPUTS = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs"
NOZX_DIR = f"{OUTPUTS}/combine_run3_100150_ctag2d_rawsf_nozx_jecshifts_automcstats_2026_09_23"
WITHZX_DIR = f"{OUTPUTS}/combine_run3_100150_ctag2d_rawsf_withzxCRmerged_noZXrate_jecshifts_automcstats_2026_09_23"
NOZX_JSON = f"{NOZX_DIR}/impacts_nozx_jecshifts_automcstats_2026_09_23.json"
WITHZX_JSON = f"{WITHZX_DIR}/impacts_withzxCRmerged_noZXrate_2026_09_23.json"
NOZX_PLOT = f"{NOZX_DIR}/impacts_noZX_final_top25_rhat_2026_09_23.pdf"
WITHZX_PLOT = f"{WITHZX_DIR}/impacts_withZX_final_top25_rhat_2026_09_23.pdf"
OUT_TABLES = "/tmp/snandaku/jesjer_results_tables.pdf"
OUT_FINAL = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/docs/hczz_ctag2d_jesjer_results.pdf"

LNN_INFO_NOZX = {
    "lumi_Run3": ("1.014", "Signal, ggZZ, qqZZ, Other_Higgs"),
    "QCDscale_gg": ("1.039", "Signal, ggZZ"),
    "pdf_gg": ("1.032", "Signal, ggZZ"),
    "kfactor_ggZZ": ("1.10", "ggZZ"),
    "BR_HZZ4l": ("1.02", "Signal, Other_Higgs"),
    "QCDscale_qqZZ": ("1.04", "qqZZ"),
}
LNN_INFO_WITHZX = dict(LNN_INFO_NOZX)
LNN_INFO_WITHZX["lumi_Run3"] = ("1.014", "Signal, ggZZ, qqZZ, Other_Higgs, ZX")
LNN_INFO_WITHZX["ZX_norm"] = ("1.30", "ZX")


def kappa_c(r):
    """Flat-direction interpretation, Felix Heyen thesis Eq. (2.29), B_SM(H->cc)=0.03."""
    return np.sqrt(0.97 * r + 0.03 * r * r)


def load_ranking(json_path, lnn_info):
    with open(json_path) as f:
        d = json.load(f)
    ranked = sorted(d["params"], key=lambda p: -abs(p.get("impact_r", 0)))
    rows = []
    for p in ranked:
        name, fit = p["name"], p["fit"]
        if name in lnn_info:
            size, procs = lnn_info[name]
            typ, val = "lnN", f"{size} ({procs})"
        elif name.startswith("prop_bin"):
            typ = "MC stat"
            if p["type"] == "Poisson":
                val = f"autoMCStats, Poisson (n_eff {fit[1]:.0f})"
            else:
                val = f"autoMCStats, pull ({fit[0]:+.3f}/{fit[2]:+.3f})"
        elif p["type"] == "Unconstrained":
            typ = "rateParam"
            val = f"{fit[1]:.2f} (+{fit[2]-fit[1]:.2f}/-{fit[1]-fit[0]:.2f}), free-floating"
        else:
            typ = "shape"
            val = f"pull +0.00 ({fit[0]:+.3f}/{fit[2]:+.3f})"
        # impact: -1 sigma / +1 sigma shifts of r-hat
        dr = (p["r"][0] - p["r"][1], p["r"][2] - p["r"][1])
        rows.append((name, typ, val, p["impact_r"], dr))
    return rows, d["POIs"][0]["fit"]


def n_named(rows):
    return sum(1 for r in rows if not r[0].startswith("prop_bin"))


def draw_table_page(pdf, rows_data, col_widths, header, title, subtitle):
    fig_h = max(11.0, 0.19 * len(rows_data) + 2.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.97)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.06, y=0.985, ha="left")
    ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, fontsize=8.5, va="top", wrap=True)
    n_rows = len(rows_data) + 1
    tbl = Table(ax, bbox=[0, 0.0, 1, 1.0 - 1.1 / fig_h])
    for j, (h, w) in enumerate(zip(header, col_widths)):
        c = tbl.add_cell(0, j, w, 1.0 / n_rows, text=h, loc="center", facecolor="#2c3e50")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
        c.get_text().set_fontsize(8)
    for i, row in enumerate(rows_data, start=1):
        bg = "#dce6f1" if i == 1 else ("#eef2f5" if i % 2 == 0 else "white")
        for j, (val, w) in enumerate(zip(row, col_widths)):
            loc = "left" if j in (1, 3) else "center"
            c = tbl.add_cell(i, j, w, 1.0 / n_rows, text=str(val), loc=loc, facecolor=bg)
            c.get_text().set_fontsize(7)
            if i == 1:
                c.get_text().set_fontweight("bold")
    tbl.auto_set_font_size(False)
    ax.add_table(tbl)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def limits_table(ax, bbox, quantiles):
    tbl = Table(ax, bbox=bbox)
    for j, h in enumerate(["Quantile", "r", "kappa_c"]):
        c = tbl.add_cell(0, j, 0.33, 1 / 6, text=h, loc="center", facecolor="#2c3e50")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
    for i, (q, r) in enumerate(quantiles, start=1):
        bg = "#dce6f1" if "median" in q else "white"
        for j, val in enumerate([q, f"{r:.4f}", f"{kappa_c(r):.2f}"]):
            c = tbl.add_cell(i, j, 0.33, 1 / 6, text=val, loc="center", facecolor=bg)
            if "median" in q:
                c.get_text().set_fontweight("bold")
    ax.add_table(tbl)


def yields_table(ax, bbox, yields):
    tbl = Table(ax, bbox=bbox)
    n = len(yields) + 1
    for j, h in enumerate(["Process", "Yield"]):
        c = tbl.add_cell(0, j, 0.5, 1 / n, text=h, loc="center", facecolor="#2c3e50")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
    for i, (p, r) in enumerate(yields, start=1):
        bg = "#dce6f1" if p == "Total bkg" else "white"
        tbl.add_cell(i, 0, 0.5, 1 / n, text=p, loc="center", facecolor=bg)
        c = tbl.add_cell(i, 1, 0.5, 1 / n, text=r, loc="center", facecolor=bg)
        if p == "Total bkg":
            c.get_text().set_fontweight("bold")
    ax.add_table(tbl)


QUANT = ["2.5%", "16%", "50% (median)", "84%", "97.5%"]
LIM_NOZX = [178.6641, 242.9138, 346.5, 510.8585, 730.3098]
LIM_WITHZX = [181.7578, 247.1201, 352.5, 522.5147, 749.9950]
STAT_NOZX = [140.7656, 191.3866, 273.0, 396.5095, 553.1971]
STAT_WITHZX = [144.6328, 195.5830, 280.5, 407.9617, 570.0557]
Y_SIG, Y_GGZZ, Y_QQZZ, Y_OH, Y_ZX = 0.053130, 6.677889, 36.318223, 55.077609, 32.604726


def main():
    nozx_rows, nozx_poi = load_ranking(NOZX_JSON, LNN_INFO_NOZX)
    withzx_rows, withzx_poi = load_ranking(WITHZX_JSON, LNN_INFO_WITHZX)
    b_nozx = Y_GGZZ + Y_QQZZ + Y_OH
    b_withzx = b_nozx + Y_ZX

    with PdfPages(OUT_TABLES) as pdf:
        # ---------------- Page 1: no-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.97)
        fig.suptitle("HcZZ ctag2d -- Results, no Z+X (2026-09-23)",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Card: combine_run3_100150_ctag2d_rawsf_nozx_jecshifts_automcstats_2026_09_23\n"
                "hplusc_mva_4class_ctag2d_scored_v8 + hplusc_mva_4class_ctag2d_jecshifts_scored_v2,\n"
                f"[100,150] GeV, MC only, 20 qqZZ-quantile bins, {n_named(nozx_rows)} named nuisances +\n"
                "autoMCStats(10). CMS_ctag2d: official up_Total/down_Total, unmodified (L0 swap correction\n"
                "reverted since 2026-09-15). Other_Higgs lhe_pdf/scalevar_muR/muF shapes included since\n"
                "2026-09-20 (only the private HPlusBottom sub-sample masked). 61.9 fb^-1, 13.6 TeV.",
                transform=ax.transAxes, fontsize=8.5, va="top")
        ax.text(0.0, 0.70, "Yields  [100,150] GeV", transform=ax.transAxes, fontsize=12, fontweight="bold")
        yields_table(ax, [0.0, 0.51, 0.55, 0.18],
                     [("Signal", f"{Y_SIG:.6f}"), ("ggZZ", f"{Y_GGZZ:.6f}"), ("qqZZ", f"{Y_QQZZ:.6f}"),
                      ("Other_Higgs", f"{Y_OH:.6f}"), ("Total bkg", f"{b_nozx:.6f}")])
        ax.text(0.0, 0.48, f"S/$\\sqrt{{B}}$ = {Y_SIG/np.sqrt(b_nozx):.4f}", transform=ax.transAxes, fontsize=9)
        ax.text(0.0, 0.42, "Expected limits (AsymptoticLimits, --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits_table(ax, [0.0, 0.22, 0.75, 0.19], list(zip(QUANT, LIM_NOZX)))
        ax.text(0.0, 0.17,
                f"Stat only (--freezeParameters allConstrainedNuisances): r={STAT_NOZX[2]:.1f} / "
                f"kappa_c={kappa_c(STAT_NOZX[2]):.2f}.\n"
                "Full syst: autoMCStats(10) does not move the no-Z+X median (346.5 with and without;\n"
                "background MC-stat ~1.3% per bin vs ~50% data stat per bin).\n"
                "kappa_c = sqrt(0.97 r + 0.03 r^2) (flat direction, F. Heyen thesis Eq. 2.29, B(H->cc)=0.03).",
                transform=ax.transAxes, fontsize=8.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 2: no-ZX systematics table ----------------
        rows = [(str(i), r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4][0]:+.1f} / {r[4][1]:+.1f}")
                for i, r in enumerate(nozx_rows, 1)]
        draw_table_page(
            pdf, rows, [0.05, 0.27, 0.08, 0.35, 0.09, 0.16],
            ["#", "Systematic", "Type", "Value (lnN size / pull)", "Impact", "dr (-1s / +1s)"],
            "Systematics, no Z+X: type, value, and impact",
            f"Ranked by impact_r (combineTool.py -M Impacts, -t -1 --expectSignal 346.5, r in [0,1500],\n"
            f"robustFit). r-hat = {nozx_poi[1]:.1f} +{nozx_poi[2]-nozx_poi[1]:.1f}/-{nozx_poi[1]-nozx_poi[0]:.1f}. "
            "MC stat = autoMCStats per-bin nuisances (prop_binhczz_bin<N>,\nN = MVA score bin 0-19).",
        )

        # ---------------- Page 4: with-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.97)
        fig.suptitle("HcZZ ctag2d -- Results with Z+X (2026-09-23)",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Card: combine_run3_100150_ctag2d_rawsf_withzxCRmerged_noZXrate_jecshifts_automcstats_2026_09_23\n"
                "Same MC/systematics as the no-Z+X card, plus Z+X from the fake-rate OS method (3P1F/2P2F)\n"
                "on the hplusc_mva_4class_CR_merged production with pkatris's WZ-subtracted OS fake rates:\n"
                "32.60 +- 4.2 events (per era 6.9/13.1/10.1/2.4), jet-inclusive (no c-jet requirement, by\n"
                "decision). Z+X normalisation: ZX_norm 30% lnN only -- the free ZX_rate rateParam was removed\n"
                "2026-09-23 (no Z+X control region in the fit; as in HIG-24-013). The datacard now applies the\n"
                "m4l window to the Z+X input itself (the earlier '133' Z+X was pkatris's inclusive-m4l yield;\n"
                "43.8 in the window).",
                transform=ax.transAxes, fontsize=8.5, va="top")
        ax.text(0.0, 0.70, "Yields in [100,150] GeV", transform=ax.transAxes, fontsize=12, fontweight="bold")
        yields_table(ax, [0.0, 0.48, 0.55, 0.21],
                     [("Signal", f"{Y_SIG:.4f}"), ("ggZZ", f"{Y_GGZZ:.4f}"), ("qqZZ", f"{Y_QQZZ:.4f}"),
                      ("Other_Higgs", f"{Y_OH:.4f}"), ("Z+X", f"{Y_ZX:.4f}"), ("Total bkg", f"{b_withzx:.4f}")])
        ax.text(0.0, 0.455, f"S/$\\sqrt{{B}}$ = {Y_SIG/np.sqrt(b_withzx):.4f}", transform=ax.transAxes, fontsize=9)
        ax.text(0.0, 0.40, "Expected limits (combine -M AsymptoticLimits -m 120 --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits_table(ax, [0.0, 0.20, 0.75, 0.19], list(zip(QUANT, LIM_WITHZX)))
        ax.text(0.0, 0.15, "Comparison", transform=ax.transAxes, fontsize=11, fontweight="bold")
        comp = [("no Z+X", LIM_NOZX[2]), ("with Z+X (free ZX_rate + ZX_norm, earlier today)", 353.0),
                ("with Z+X (ZX_norm only, current)", LIM_WITHZX[2])]
        tbl = Table(ax, bbox=[0.0, 0.0, 0.95, 0.14])
        for j, h in enumerate(["Configuration", "r (median)", "kappa_c"]):
            w = 0.6 if j == 0 else 0.2
            c = tbl.add_cell(0, j, w, 0.25, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (cfg, r) in enumerate(comp, start=1):
            bg = "#dce6f1" if i == len(comp) else "white"
            tbl.add_cell(i, 0, 0.6, 0.25, text=cfg, loc="left", facecolor=bg)
            tbl.add_cell(i, 1, 0.2, 0.25, text=f"{r:.1f}", loc="center", facecolor=bg)
            tbl.add_cell(i, 2, 0.2, 0.25, text=f"{kappa_c(r):.2f}", loc="center", facecolor=bg)
        ax.add_table(tbl)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 5: with-ZX systematics table ----------------
        rows = [(str(i), r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4][0]:+.1f} / {r[4][1]:+.1f}")
                for i, r in enumerate(withzx_rows, 1)]
        draw_table_page(
            pdf, rows, [0.05, 0.27, 0.08, 0.35, 0.09, 0.16],
            ["#", "Systematic", "Type", "Value (lnN size / pull)", "Impact", "dr (-1s / +1s)"],
            "Systematics with Z+X: type, value, and impact",
            f"Ranked by impact_r (-t -1 --expectSignal 352.5, r in [0,1500], robustFit). r-hat = "
            f"{withzx_poi[1]:.1f} +{withzx_poi[2]-withzx_poi[1]:.1f}/-{withzx_poi[1]-withzx_poi[0]:.1f}.\n"
            "ZX_norm is one-sided (-38/+5): inherited from the one-sided CMS_ctag2d (see last page).",
        )

        # ---------------- Page 7: stat-only comparison ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.97)
        fig.suptitle("Stat-only limits: with vs. without Z+X",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "combine -M AsymptoticLimits --freezeParameters allConstrainedNuisances on the current cards\n"
                "(all systematics, ZX_norm and the autoMCStats parameters frozen; no free parameter other than r).",
                transform=ax.transAxes, fontsize=8.5, va="top")
        tbl = Table(ax, bbox=[0.0, 0.62, 1.0, 0.28])
        headers = ["Quantile", "r (no ZX)", "kc (no ZX)", "r (with ZX)", "kc (with ZX)", "dr", "dr %"]
        widths = [0.20, 0.15, 0.13, 0.15, 0.13, 0.12, 0.12]
        for j, h in enumerate(headers):
            c = tbl.add_cell(0, j, widths[j], 1 / 6, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
            c.get_text().set_fontsize(8)
        for i, (q, a, b) in enumerate(zip(QUANT, STAT_NOZX, STAT_WITHZX), start=1):
            bg = "#dce6f1" if "median" in q else "white"
            vals = [q, f"{a:.4f}", f"{kappa_c(a):.2f}", f"{b:.4f}", f"{kappa_c(b):.2f}",
                    f"{b-a:+.2f}", f"{100*(b-a)/a:+.1f}%"]
            for j, val in enumerate(vals):
                c = tbl.add_cell(i, j, widths[j], 1 / 6, text=val, loc="center", facecolor=bg)
                c.get_text().set_fontsize(8)
                if "median" in q:
                    c.get_text().set_fontweight("bold")
        ax.add_table(tbl)
        ax.text(0.0, 0.56,
                f"Stat only, Z+X loosens the median by {100*(STAT_WITHZX[2]-STAT_NOZX[2])/STAT_NOZX[2]:.1f}% "
                f"({STAT_NOZX[2]:.1f} -> {STAT_WITHZX[2]:.1f});\n"
                f"with full systematics by {100*(LIM_WITHZX[2]-LIM_NOZX[2])/LIM_NOZX[2]:.1f}% "
                f"({LIM_NOZX[2]:.1f} -> {LIM_WITHZX[2]:.1f}).\n"
                f"Systematics loosen the median by {100*(LIM_NOZX[2]-STAT_NOZX[2])/STAT_NOZX[2]:.0f}% (no Z+X) / "
                f"{100*(LIM_WITHZX[2]-STAT_WITHZX[2])/STAT_WITHZX[2]:.0f}% (with Z+X).",
                transform=ax.transAxes, fontsize=9, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 8: ctag2d -> ZX_norm test ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.97)
        fig.suptitle("What drives the Z+X impact? (with-Z+X card)",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Impacts from Asimov fits with r injected at each variant's median; dr for -1 sigma / +1 sigma.",
                transform=ax.transAxes, fontsize=8.5, va="top")
        test = [("Free ZX_rate + ZX_norm (before 09-23)", "353.0", "-0.9 / +0.1", "+57 / -163", "ZX_rate -82.9 / +4.9"),
                ("A: production (ZX_norm only)", "352.5", "-38.2 / +5.0", "+57 / -163", "--"),
                ("B: CMS_ctag2d frozen", "337.5", "-3.6 / -1.7", "frozen", "--"),
                ("C: CMS_ctag2d Down = mirror of Up", "528.0", "-46.9 / +41.0", "+298 / -208", "--")]
        tbl = Table(ax, bbox=[0.0, 0.70, 1.0, 0.20])
        headers = ["Variant", "median r", "ZX_norm impact", "CMS_ctag2d impact", "other Z+X NP"]
        widths = [0.36, 0.11, 0.16, 0.17, 0.20]
        for j, h in enumerate(headers):
            c = tbl.add_cell(0, j, widths[j], 1 / 5, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
            c.get_text().set_fontsize(8)
        for i, row in enumerate(test, start=1):
            bg = "#dce6f1" if row[0].startswith("A:") else "white"
            for j, val in enumerate(row):
                c = tbl.add_cell(i, j, widths[j], 1 / 5, text=val, loc="left" if j == 0 else "center", facecolor=bg)
                c.get_text().set_fontsize(8)
        ax.add_table(tbl)
        ax.text(0.0, 0.66,
                "B: without CMS_ctag2d the ZX_norm impact nearly vanishes -- its size and asymmetry come from the\n"
                "fit compensating a Z+X shift through CMS_ctag2d. A ZX_norm scan with ctag2d frozen is a shallow\n"
                "hump (r-hat 346.9 ... 352.5 ... 349.0 over +-1.1 sigma), hence the same-sign -3.6/-1.7.\n\n"
                "C: a symmetric CMS_ctag2d makes ZX_norm symmetric, so the asymmetry is inherited from ctag2d\n"
                "(Up: Signal +36.5%, Other_Higgs +15.3%, qqZZ +8.6%; Down: -3.1%, -4.4%, +0.3%). C is a\n"
                "diagnostic only, not a valid model.\n\n"
                "Mechanism: Z+X down -> the fit refills the low-score bins via ctag2d Up (Other_Higgs +29% in\n"
                "bins 0-9), which also raises the signal yield -> r-hat down. Z+X up needs ctag2d Down, which\n"
                "is ~flat -> other NPs absorb it -> r-hat ~unchanged.\n\n"
                "Rebinning (3-20 bins, merged) does not change the ZX_norm or CMS_ctag2d impacts.\n"
                "Open: the one-sided CMS_ctag2d Up (L0 up/down, with the ctag2d calibration contact).",
                transform=ax.transAxes, fontsize=9, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote table pages: {OUT_TABLES}")

    tables = PdfReader(OUT_TABLES)
    writer = PdfWriter()
    writer.add_page(tables.pages[0])                       # 1: no-ZX yields/limits
    writer.add_page(tables.pages[1])                       # 2: no-ZX systematics table
    writer.add_page(PdfReader(NOZX_PLOT).pages[0])         # 3: no-ZX impacts (top 25)
    writer.add_page(tables.pages[2])                       # 4: with-ZX yields/limits
    writer.add_page(tables.pages[3])                       # 5: with-ZX systematics table
    writer.add_page(PdfReader(WITHZX_PLOT).pages[0])       # 6: with-ZX impacts (top 25)
    writer.add_page(tables.pages[4])                       # 7: stat-only comparison
    writer.add_page(tables.pages[5])                       # 8: ctag2d -> ZX_norm test
    with open(OUT_FINAL, "wb") as f:
        writer.write(f)
    print(f"Wrote final 8-page PDF: {OUT_FINAL}")


if __name__ == "__main__":
    main()
