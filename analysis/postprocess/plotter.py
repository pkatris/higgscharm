import re
import yaml
import hist
import logging
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import ticker
from matplotlib.lines import Line2D
from coffea.processor import accumulate
from hist.intervals import poisson_interval
from matplotlib.offsetbox import AnchoredText
from analysis.filesets.utils import get_workflow_key_process_map, get_process_era_map
from analysis.histograms import VariableAxis, IntegerAxis
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.postprocess.utils import (
    setup_logger,
    divide_by_binwidth,
    get_variations_keys,
)

np.seterr(invalid="ignore")
np.seterr(divide="ignore")


class CoffeaPlotter:
    def __init__(
        self,
        workflow: str,
        year: str,
        processed_histograms: dict,
        output_dir: str,
        group_by: str,
        pass_axis: str,
        rebin: dict = None,
    ):
        self.workflow = workflow
        self.year = year
        self.processed_histograms = processed_histograms
        self.output_dir = output_dir
        self.group_by = group_by
        self.pass_axis = pass_axis
        # {variable: int factor | list of new bin edges}, applied at plotting time only
        self.rebin = rebin or {}

        # get histogram config
        config_builder = WorkflowConfigBuilder(workflow=workflow)
        workflow_config = config_builder.build_workflow_config()
        self.histogram_config = workflow_config.histogram_config
        self.datasets = workflow_config.datasets

        # load luminosities and plotting style params
        postprocess_dir = Path.cwd() / "analysis" / "postprocess"
        style_file = postprocess_dir / "style.yaml"
        luminosity_file = postprocess_dir / "luminosity.yaml"
        color_map_file = postprocess_dir / "process_color_map.yaml"

        with open(style_file, "r") as f:
            self.style = yaml.safe_load(f)
        with open(f"{Path.cwd()}/analysis/postprocess/luminosity.yaml", "r") as f:
            self.luminosities = yaml.safe_load(f)

        # set processes -> era map
        self.process_era_map = get_process_era_map(year)

        # set processes -> color map
        with open(color_map_file, "r") as f:
            color_map = yaml.safe_load(f)
        self.color_map = {p: c for p, c in color_map.items()}

    def get_histogram(
        self,
        variable,
        variation,
        category,
        histogram,
        other_category=None,
    ):
        """returns histogram by processes/variable/category"""
        # get variable histogram for nominal variation and category
        selector = {"variation": variation}
        if "category" in histogram.axes.name:
            selector["category"] = category
        if self.pass_axis:
            selector[self.pass_axis] = True
        if other_category is not None:
            selector[self.group_by["name"]] = other_category
        histogram = histogram[selector].project(variable)
        histogram = self.apply_rebin(histogram, variable)
        # if axis type is variable divide by bin width
        if isinstance(self.histogram_config.axes[variable], VariableAxis):
            histogram = divide_by_binwidth(histogram)
        elif self.per_gev(variable):
            histogram = self.scale_by_binwidth(histogram)
        return histogram

    # ------------------------------------------------------------------
    # rebinning (plotting time only; processed histograms are untouched)
    # ------------------------------------------------------------------
    def per_gev(self, variable):
        """True if the plotted y-axis is per unit of x (uneven bins)"""
        return isinstance(self.histogram_config.axes[variable], VariableAxis) or isinstance(
            self.rebin.get(variable), (list, tuple)
        )

    @staticmethod
    def scale_by_binwidth(histogram):
        widths = histogram.axes[0].widths
        out = histogram.copy()
        view = out.view(flow=False)
        if hasattr(view, "value"):
            view.value[...] = view.value / widths
            view.variance[...] = view.variance / widths**2
        else:
            view[...] = view / widths
        return out

    def apply_rebin(self, histogram, variable):
        """rebin a 1D histogram by an integer factor or onto new edges"""
        spec = self.rebin.get(variable)
        if spec is None:
            return histogram
        axis = histogram.axes[0]
        old_edges = np.asarray(axis.edges)
        if isinstance(spec, int):
            if spec < 1 or len(axis) % spec != 0:
                raise ValueError(
                    f"rebin factor {spec} for '{variable}' must divide its {len(axis)} bins "
                    f"(possible: {[k for k in range(2, len(axis) + 1) if len(axis) % k == 0]})"
                )
            new_edges = old_edges[::spec]
        else:
            new_edges = np.asarray(spec, dtype=float)
            missing = [e for e in new_edges if not np.isclose(old_edges, e).any()]
            if missing or np.any(np.diff(new_edges) <= 0):
                raise ValueError(
                    f"rebin edges for '{variable}' must be increasing and a subset of the "
                    f"original edges (not original edges: {missing})"
                )
        # map every original bin (incl. flow) onto the new binning
        centers = 0.5 * (old_edges[1:] + old_edges[:-1])
        idx = np.searchsorted(new_edges, centers, side="right")  # 0 = underflow, len = overflow
        idx = np.concatenate([[0], idx, [len(new_edges)]])
        new_axis = hist.axis.Variable(new_edges, name=axis.name, label=axis.label)
        old_view = histogram.view(flow=True)
        if hasattr(old_view, "value"):
            new = hist.Hist(new_axis, storage=hist.storage.Weight())
            vals = np.bincount(idx, weights=old_view.value, minlength=len(new_edges) + 1)
            vars_ = np.bincount(idx, weights=old_view.variance, minlength=len(new_edges) + 1)
            new_view = new.view(flow=True)
            new_view.value[...] = vals
            new_view.variance[...] = vars_
        else:
            new = hist.Hist(new_axis, storage=hist.storage.Double())
            new.view(flow=True)[...] = np.bincount(
                idx, weights=old_view, minlength=len(new_edges) + 1
            )
        return new

    def get_variations(
        self,
        variable,
        category,
        variation,
        histogram,
    ):
        """returns variation histogram by processes/variable/category"""
        # get variable histogram for nominal variation and category
        selectorup = {"variation": f"{variation}Up"}
        selectordown = {"variation": f"{variation}Down"}
        if self.pass_axis:
            selectorup[self.pass_axis] = True
            selectordown[self.pass_axis] = True
        if "category" in histogram.axes.name:
            selectorup["category"] = category
            selectordown["category"] = category
        histogram_up = self.apply_rebin(histogram[selectorup].project(variable), variable)
        histogram_down = self.apply_rebin(histogram[selectordown].project(variable), variable)
        # if axis type is variable divide by bin width
        if isinstance(self.histogram_config.axes[variable], VariableAxis):
            histogram_up = divide_by_binwidth(histogram_up)
            histogram_down = divide_by_binwidth(histogram_down)
        elif self.per_gev(variable):
            histogram_up = self.scale_by_binwidth(histogram_up)
            histogram_down = self.scale_by_binwidth(histogram_down)
        return histogram_up, histogram_down

    @staticmethod
    def blind_histogram(histogram, blind_range):
        """
        return a copy of a 1D histogram with every bin overlapping
        [low, high] set to NaN (NaN points are not drawn by matplotlib,
        so they disappear from both the main panel and the ratio panel)
        """
        if blind_range is None:
            return histogram
        low, high = blind_range
        blinded = histogram.copy()
        edges = blinded.axes[0].edges
        # any bin overlapping the window is blinded (conservative at the edges)
        in_window = (edges[:-1] < high) & (edges[1:] > low)
        view = blinded.view(flow=False)
        if hasattr(view, "value"):
            # Weight storage: blind both sum of weights and variance
            view.value[in_window] = np.nan
            view.variance[in_window] = np.nan
        else:
            # Double/Int storage
            view[in_window] = np.nan
        return blinded

    def collect_histograms_for_plotting(self, variable, category):
        histogram_info = {}
        if "mc" in self.datasets:
            histogram_info["mc"] = {"nominal": {}, "variations": {}}
        if "signal" in self.datasets:
            histogram_info["signal"] = {"nominal": {}}
        if self.group_by != "process":
            histogram_info["categories"] = {}

        for process, histogram_dict in self.processed_histograms.items():
            if variable in histogram_dict:
                aux_histogram = histogram_dict[variable]
            else:
                for key in histogram_dict:
                    if variable in histogram_dict[key].axes.name:
                        aux_histogram = histogram_dict[key]
                        break

            if process == "Data":
                histogram_info["data"] = self.get_histogram(
                    variable=variable,
                    category=category,
                    variation="nominal",
                    histogram=aux_histogram,
                )
            else:
                key = self.process_era_map[process]
                if self.group_by != "process":
                    cat_axis = aux_histogram.axes[self.group_by["name"]]
                    self.category_map = {cat_axis.index(cat): cat for cat in cat_axis}
                    group_by_categories = [cat_axis.index(cat) for cat in cat_axis]
                    for cat in group_by_categories:
                        if cat not in histogram_info["categories"]:
                            histogram_info["categories"][cat] = {}
                        histogram_info["categories"][cat][process] = self.get_histogram(
                            variable=variable,
                            category=category,
                            variation="nominal",
                            other_category=cat,
                            histogram=aux_histogram,
                        )
                else:
                    histogram_info[key]["nominal"][process] = self.get_histogram(
                        variable=variable,
                        category=category,
                        variation="nominal",
                        histogram=aux_histogram,
                    )

                # save variations histograms
                if key == "mc":
                    var_cats = set(aux_histogram.axes["variation"])
                    process_nominal = None
                    for variation in get_variations_keys(self.processed_histograms):
                        if f"{variation}Up" in var_cats and f"{variation}Down" in var_cats:
                            up, down = self.get_variations(
                                variable=variable,
                                category=category,
                                variation=variation,
                                histogram=aux_histogram,
                            )
                        else:
                            # FIX 2026-10-01: a process that does not carry this
                            # variation (e.g. kfactor_ggzz_* for qqZZ, scalevar_* for
                            # the MCFM ggZZ samples) is unaffected by it, so it must
                            # enter the varied total at its NOMINAL. Previously it was
                            # skipped (`continue`), i.e. counted as 0, which made the
                            # varied total far below nominal for both Up and Down and
                            # inflated the lower edge of the band by tens of percent.
                            if process_nominal is None:
                                process_nominal = self.get_histogram(
                                    variable=variable,
                                    category=category,
                                    variation="nominal",
                                    histogram=aux_histogram,
                                )
                            up, down = process_nominal.copy(), process_nominal.copy()
                        if f"{variation}Up" in histogram_info[key]["variations"]:
                            histogram_info[key]["variations"][f"{variation}Up"] += up
                            # FIX: previously accumulated `up` into the Down variation
                            histogram_info[key]["variations"][f"{variation}Down"] += down
                        else:
                            histogram_info[key]["variations"][f"{variation}Up"] = up
                            histogram_info[key]["variations"][f"{variation}Down"] = down

        return histogram_info

    def plot_uncert_band(self, histogram_info, ax):
        # initialize up/down errors with statisticall error
        # FIX: copy the arrays; previously err2_up, err2_down and
        # self.nominal_variances were the same object, so every `+=` below
        # modified all three and the up/down bands came out identical
        mcstat_err2 = np.array(self.nominal_variances, dtype=float)
        err2_up = mcstat_err2.copy()
        err2_down = mcstat_err2.copy()

        for variation in get_variations_keys(self.processed_histograms):
            # Up/down variations for a single MC sample
            var_up = histogram_info["mc"]["variations"][f"{variation}Up"].values()
            var_down = histogram_info["mc"]["variations"][f"{variation}Down"].values()
            # Compute the uncertainties corresponding to the up/down variations
            err_up = var_up - self.nominal_values
            err_down = var_down - self.nominal_values
            # Compute the flags to check which of the two variations (up and down) are pushing the nominal value up and down
            up_is_up = err_up > 0
            down_is_down = err_down < 0
            # Compute the flag to check if the uncertainty is one-sided, i.e. when both variations are up or down
            is_onesided = up_is_up ^ down_is_down
            # Sum in quadrature of the systematic uncertainties taking into account if the uncertainty is one- or double-sided
            err2_up_twosided = np.where(up_is_up, err_up**2, err_down**2)
            err2_down_twosided = np.where(up_is_up, err_down**2, err_up**2)
            err2_max = np.maximum(err2_up_twosided, err2_down_twosided)
            err2_up_onesided = np.where(is_onesided & up_is_up, err2_max, 0)
            err2_down_onesided = np.where(is_onesided & down_is_down, err2_max, 0)
            err2_up_combined = np.where(is_onesided, err2_up_onesided, err2_up_twosided)
            err2_down_combined = np.where(
                is_onesided, err2_down_onesided, err2_down_twosided
            )
            # Sum in quadrature of the systematic uncertainty corresponding to a MC sample
            err2_up += err2_up_combined
            err2_down += err2_down_combined

        self.band_up = self.nominal_values + np.sqrt(err2_up)
        self.band_down = self.nominal_values - np.sqrt(err2_down)
        # plot stat + syst uncertainty band
        ax.bar(
            x=self.centers,
            height=self.band_up - self.band_down,
            width=self.widths,
            bottom=self.band_down,
            **self.style["uncert_band_kwargs"],
        )

    def plot_ratio(self, rax):
        # compute Data/MC ratio
        # (blinded bins are NaN in self.data_values, so their ratio points are not drawn)
        num = self.data_values
        den = self.nominal_values
        ratio = num / den
        # only the uncertainty of num (DATA) propagated
        num_variances = self.data_variances
        ratio_variance = num_variances * np.power(den, -2)
        ratio_uncert = np.abs(poisson_interval(ratio, ratio_variance) - ratio)
        # plot ratio and x-y errors
        xerr = self.edges[1:] - self.edges[:-1]
        rax.errorbar(
            self.centers,
            ratio,
            xerr=xerr / 2,
            yerr=ratio_uncert,
            **self.style["ratio_error_kwargs"],
        )
        # plot ratio uncertainty band
        ratio_up = np.concatenate([[0], self.band_up / den])
        ratio_down = np.concatenate([[0], self.band_down / den])
        ratio_up[np.isnan(ratio_up)] = 1.0
        ratio_down[np.isnan(ratio_down)] = 1.0
        ratio_uncertainty_band = rax.fill_between(
            self.edges,
            ratio_up,
            ratio_down,
            step="pre",
            **self.style["uncert_band_kwargs"],
        )
        # plot horizontal reference line at 1
        xmin, xmax = rax.get_xlim()
        rax.hlines(1, xmin, xmax, color="k", linestyle=":")

    def add_text(self, variable, category, ax):
        text_map = {
            "ztoee": rf"$ Z \rightarrow ee$ events",
            "ztomumu": rf"$ Z \rightarrow \mu \mu$ events",
        }
        if self.workflow.startswith("zplusl_"):
            zplusl_method = {
                "zplusl_os": "(OS)",
                "zplusl_ss": "(SS)",
            }
            zplusl_text = {
                "electron": rf"$Z+e$ {zplusl_method[self.workflow]} events",
                "muon": rf"$Z+\mu$ {zplusl_method[self.workflow]} events",
            }
            if self.pass_axis:
                zplusl_text = {
                    "electron": f"$Z+e$ {zplusl_method[self.workflow]} events with $e$ passing selection",
                    "muon": f"$Z+\mu$ {zplusl_method[self.workflow]} events with $\mu$ passing selection",
                }
            zplusl_text_map = {
                "zplusl_os": zplusl_text[category],
                "zplusl_ss": zplusl_text[category],
                "zplusl_ss_maximal": zplusl_text[category] + " (maximal)",
                "zplusl_ss_minimal": zplusl_text[category] + " (minimal)",
                "zplusl_ss_intermediate": zplusl_text[category] + " (intermediate)",
            }
            text_map = {**text_map, **zplusl_text_map}

        if self.workflow.startswith("zplusll"):
            zplusll_text_map = {
                "zplusll_os": (
                    "3P+1F control sample"
                    if "1fcr" in variable
                    else "2P+2F control sample"
                ),
                "zplusll_ss": "SS-SF control sample",
            }
            text_map = {**text_map, **zplusll_text_map}

        if self.workflow == "zzto4l":
            zzto4l_identifier = {
                "zz_mass_4e": r"$4e$",
                "zz_mass_4mu": r"$4\mu$",
                "zz_mass_2mu2e": r"$2\mu 2e$",
                "zz_mass_2e2mu": r"$2e2\mu$",
            }
            zzto4l_text = (
                r"$ZZ\rightarrow$"
                + zzto4l_identifier.get(variable, r"$4\ell$")
                + " events"
            )
            zzto4l_text_map = {"zzto4l": zzto4l_text}
            text_map = {**text_map, **zzto4l_text_map}

        if self.workflow == "hplusc":
            hplusc_test = {"hplusc": "H(ZZ)+c-jet events"}
            text_map = text_map = {**text_map, **hplusc_test}

        ax.add_artist(
            AnchoredText(
                text_map.get(self.workflow, f"{self.workflow} events") + "\n",
                loc="upper left",
                frameon=False,
            )
        )

    def add_xylabels(self, variable, category, add_ratio, ax, rax):
        ylabel = "Events"
        m = re.search(r"\[(.+?)\]", self.histogram_config.axes[variable].label or "")
        unit = m.group(1) if m else ""
        if self.per_gev(variable):
            ylabel += f" / {unit}" if unit else " / unit"
        elif isinstance(self.rebin.get(variable), int):
            width = self.edges[1] - self.edges[0]
            ylabel += f" / {width:g} {unit}".rstrip()

        xlabel = self.histogram_config.axes[variable].label

        if self.workflow.startswith("zplusl_"):
            if category == "electron":
                xlabel = xlabel.replace(r"\ell", r"e")
            elif category == "muon":
                xlabel = xlabel.replace(r"\ell", r"\mu")

        if add_ratio:
            ax.set(xlabel=None, ylabel=ylabel)
            rax.set(
                xlabel=xlabel,
                ylabel="Data / Pred",
                facecolor="white",
            )
        else:
            ax.set(xlabel=xlabel, ylabel=ylabel)

    def plot_histograms(
        self,
        variable: str,
        category: str,
        yratio_limits: str = None,
        log: bool = False,
        add_ratio: bool = True,
        blind: bool = False,
        extension: str = "png",
        blind_range: tuple = None,
        signal_scales: dict = None,
    ):
        """
        signal_scales: optional {signal_process: factor} used to scale signal
        overlays for visibility, e.g. {"somesmsignal": 100, "hb": 100}.
        Only the drawn signal lines are scaled (and their legend labels get
        "x factor"); the MC stack, uncertainty band and ratio are untouched.
        """
        signal_scales = signal_scales or {}
        if blind:
            add_ratio = False

        setup_logger(self.output_dir)

        # set plot params
        hep.style.use(hep.style.CMS)
        plt.rcParams.update(self.style["rcParams"])

        # get mc, data and signal histograms
        histogram_info = self.collect_histograms_for_plotting(variable, category)

        # get nominal MC histograms
        mc_colors, mc_labels = [], []
        if self.group_by == "process":
            if (self.workflow in ["zzto4l", "hplusc"]) and ("zz_mass" in variable):
                mc_labels = ["ggToZZ", "qqToZZ", "H(125)"]
                nominal_mc_hists = [
                    histogram_info["mc"]["nominal"][p] for p in mc_labels
                ]
                mc_colors = [self.color_map[p] for p in mc_labels]
            else:
                nominal_mc_hists = list(histogram_info["mc"]["nominal"].values())
                for process in histogram_info["mc"]["nominal"]:
                    mc_labels.append(process)
                    mc_colors.append(self.color_map[process])
        else:
            nominal_mc_hists = []
            for cat in histogram_info["categories"]:
                nominal_mc_hists.append(
                    accumulate(histogram_info["categories"][cat].values())
                )
                labels_map = {v: k for k, v in self.group_by["label"].items()}
                mc_labels.append(labels_map[self.category_map[cat]])

        mc_histogram = accumulate(nominal_mc_hists)
        self.nominal_values = mc_histogram.values()
        self.nominal_variances = mc_histogram.variances()
        self.edges = mc_histogram.axes.edges[0]
        self.centers = mc_histogram.axes.centers[0]
        self.widths = mc_histogram.axes.widths[0]

        # get variation MC histograms
        variation_histograms = histogram_info["mc"]["variations"]

        # get Data histogram (blinded in blind_range, if requested)
        if "data" not in self.datasets:
            blind = True
        if not blind:
            data_histogram = self.blind_histogram(histogram_info["data"], blind_range)
            self.data_values = data_histogram.values()
            self.data_variances = data_histogram.variances()

        # plot stacked MC and Data histograms
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9, 10),
            tight_layout=True,
            gridspec_kw={"height_ratios": (4, 1)},
            sharex=True,
        )
        if not add_ratio:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(9, 9),
                tight_layout=True,
            )
        mc_hist_args = {
            "H": nominal_mc_hists,
            "label": mc_labels,
            "flow": "none",
            "ax": ax,
        }
        mc_hist_args.update(self.style["mc_hist_kwargs"])
        if mc_colors:
            mc_hist_args.update({"color": mc_colors})
        if (self.workflow in ["zzto4l", "hplusc"]) and ("zz_mass" in variable):
            mc_hist_args["sort"] = None
        hep.histplot(**mc_hist_args)
        if not blind:
            hep.histplot(
                data_histogram,
                label="Data",
                flow="none",
                ax=ax,
                **self.style["data_hist_kwargs"],
            )
        if "signal" in self.datasets:
            signal_hists = histogram_info["signal"]["nominal"]
            missing = [p for p in signal_scales if p not in signal_hists]
            if missing:
                logging.warning(
                    f"signal_scales: {missing} not found among signal processes "
                    f"{list(signal_hists)}; they will not be scaled"
                )
            for signal_process, signal_histogram in signal_hists.items():
                label = signal_process
                factor = signal_scales.get(signal_process)
                if factor is not None and factor != 1:
                    # hist supports scalar multiplication (values *f, variances *f^2)
                    signal_histogram = signal_histogram * factor
                    label = f"{signal_process} $\\times$ {factor:g}"
                hep.histplot(
                    signal_histogram,
                    label=label,
                    color=self.color_map[signal_process],
                    flow="none",
                    ax=ax,
                    **self.style["signal_hist_kwargs"],
                )

        # plot uncertainty band
        self.plot_uncert_band(histogram_info, ax)

        # plot ratio
        if add_ratio:
            self.plot_ratio(rax)

        # shade the blinded window
        if (blind_range is not None) and (not blind):
            blind_span_kwargs = {
                "facecolor": "none",
                "edgecolor": "gray",
                "hatch": "///",
                "linewidth": 0,
                "zorder": 0,
            }
            ax.axvspan(*blind_range, label="Blinded", **blind_span_kwargs)
            if add_ratio:
                rax.axvspan(*blind_range, **blind_span_kwargs)

        # set limits
        hist_edges = np.array([[i, j] for i, j in zip(self.edges[:-1], self.edges[1:])])
        xlimits = np.min(hist_edges[self.nominal_values > 0]), np.max(
            hist_edges[self.nominal_values > 0]
        )
        ax.set_xlim(xlimits)
        if add_ratio:
            rax.set_xlim(xlimits)
            rax.set_ylim(yratio_limits)

        # set axes labels
        self.add_xylabels(variable, category, add_ratio, ax, rax)

        # set plot text
        self.add_text(variable, category, ax)

        # set log scale
        if log:
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_yscale("log")
            ax.set_ylim(top=np.max(mc_histogram.values()) * 100)
        else:
            ylim = ax.get_ylim()[1]
            ax.set_ylim(0, ylim + 0.2 * ylim)
        ax.legend(
            frameon=True,
            loc="upper right",
            fontsize=13,
        )
        if isinstance(self.histogram_config.axes[variable], IntegerAxis):
            start = self.histogram_config.axes[variable].start
            stop = self.histogram_config.axes[variable].stop
            categories = np.arange(start, stop)
            if len(categories) > 20:
                for i, label in enumerate(rax.get_xticklabels()):
                    if i % 5 != 0:  # Show only every 5th tick
                        label.set_visible(False)
        # add CMS info
        lumi = self.luminosities[self.year] * 1e-3
        energy = "13" if self.year.startswith("201") else "13.6"
        hep.cms.lumitext(
            f"{lumi:.1f} fb$^{{-1}}$ ({self.year}, {energy} TeV)",
            ax=ax,
        )
        hep.cms.text("Preliminary", ax=ax)
        # save histograms
        output_path = Path(f"{self.output_dir}/{category}")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        figname = f"{str(output_path)}/{self.workflow}_{category}_{variable}_{self.year}.{extension}"
        if self.pass_axis:
            figname = f"{str(output_path)}/{self.workflow}_{category}_{variable}_{self.pass_axis}_{self.year}.{extension}"
        if self.group_by != "process":
            figname = f"{str(output_path)}/{self.workflow}_{category}_{variable}_{self.year}_groupedby{self.group_by['name']}.{extension}"
        fig.savefig(figname)
        plt.close()

    def plot_fake_rate(self, category, ylim=(None, 0.35), extension="pdf"):
        edges = (
            self.processed_histograms["Data"]["probe_lepton"]
            .project("probe_lepton_pt")
            .axes.edges[0]
        )
        xerr = edges[1:] - edges[:-1]
        centers = (
            self.processed_histograms["Data"]["probe_lepton"]
            .project("probe_lepton_pt")
            .axes.centers[0]
        )

        data_barrel = self.processed_histograms["Data"]["probe_lepton"][
            {"variation": "nominal", "category": category, "is_barrel_lepton": True}
        ].project("probe_lepton_pt", "is_passing_lepton")
        data_endcap = self.processed_histograms["Data"]["probe_lepton"][
            {"variation": "nominal", "category": category, "is_barrel_lepton": False}
        ].project("probe_lepton_pt", "is_passing_lepton")
        wz_barrel = self.processed_histograms["WZ"]["probe_lepton"][
            {"variation": "nominal", "category": category, "is_barrel_lepton": True}
        ].project("probe_lepton_pt", "is_passing_lepton")
        wz_endcap = self.processed_histograms["WZ"]["probe_lepton"][
            {"variation": "nominal", "category": category, "is_barrel_lepton": False}
        ].project("probe_lepton_pt", "is_passing_lepton")
        data_wz_barrel = data_barrel + (wz_barrel * -1)
        data_wz_endcap = data_endcap + (wz_endcap * -1)

        data_barrel_ratio = (
            data_barrel[{"is_passing_lepton": True}].values()
            / data_barrel[{"is_passing_lepton": sum}].values()
        )
        data_endcap_ratio = (
            data_endcap[{"is_passing_lepton": True}].values()
            / data_endcap[{"is_passing_lepton": sum}].values()
        )
        data_wz_barrel_ratio = (
            data_wz_barrel[{"is_passing_lepton": True}].values()
            / data_wz_barrel[{"is_passing_lepton": sum}].values()
        )
        data_wz_endcap_ratio = (
            data_wz_endcap[{"is_passing_lepton": True}].values()
            / data_wz_endcap[{"is_passing_lepton": sum}].values()
        )

        data_barrel_den = data_barrel[{"is_passing_lepton": sum}].values()
        data_barrel_num_variances = data_barrel[{"is_passing_lepton": True}].variances()
        data_barrel_ratio_variance = data_barrel_num_variances * np.power(
            data_barrel_den, -2
        )
        data_barrel_ratio_uncert = np.abs(
            poisson_interval(data_barrel_ratio, data_barrel_ratio_variance)
            - data_barrel_ratio
        )

        data_endcap_den = data_endcap[{"is_passing_lepton": sum}].values()
        data_endcap_num_variances = data_endcap[{"is_passing_lepton": True}].variances()
        data_endcap_ratio_variance = data_endcap_num_variances * np.power(
            data_endcap_den, -2
        )
        data_endcap_ratio_uncert = np.abs(
            poisson_interval(data_endcap_ratio, data_endcap_ratio_variance)
            - data_endcap_ratio
        )

        data_wz_barrel_den = data_wz_barrel[{"is_passing_lepton": sum}].values()
        data_wz_barrel_num_variances = data_wz_barrel[
            {"is_passing_lepton": True}
        ].variances()
        data_wz_barrel_ratio_variance = data_wz_barrel_num_variances * np.power(
            data_wz_barrel_den, -2
        )
        data_wz_barrel_ratio_uncert = np.abs(
            poisson_interval(data_wz_barrel_ratio, data_wz_barrel_ratio_variance)
            - data_wz_barrel_ratio
        )

        data_wz_endcap_den = data_wz_endcap[{"is_passing_lepton": sum}].values()
        data_wz_endcap_num_variances = data_wz_endcap[
            {"is_passing_lepton": True}
        ].variances()
        data_wz_endcap_ratio_variance = data_wz_endcap_num_variances * np.power(
            data_wz_endcap_den, -2
        )
        data_wz_endcap_ratio_uncert = np.abs(
            poisson_interval(data_wz_endcap_ratio, data_wz_endcap_ratio_variance)
            - data_wz_endcap_ratio
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        errorbar_container1 = ax.errorbar(
            centers,
            data_barrel_ratio,
            xerr=xerr / 2,
            yerr=data_barrel_ratio_uncert,
            fmt="bo",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        errorbar_container2 = ax.errorbar(
            centers,
            data_endcap_ratio,
            xerr=xerr / 2,
            yerr=data_endcap_ratio_uncert,
            fmt="ro",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        # Customizing error lines after creation (optional for full control)
        for bar in errorbar_container1[2]:
            bar.set_linestyle("-")
            bar.set_color("b")
            bar.set_linewidth(2)
        for bar in errorbar_container2[2]:
            bar.set_linestyle("-")
            bar.set_color("r")
            bar.set_linewidth(2)

        errorbar_container3 = ax.errorbar(
            centers,
            data_wz_barrel_ratio,
            xerr=xerr / 2,
            yerr=data_wz_barrel_ratio_uncert,
            fmt="bo",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        errorbar_container4 = ax.errorbar(
            centers,
            data_wz_endcap_ratio,
            xerr=xerr / 2,
            yerr=data_wz_endcap_ratio_uncert,
            fmt="ro",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        # Customizing error lines after creation (optional for full control)
        for bar in errorbar_container3[2]:
            bar.set_linestyle(":")
            bar.set_color("b")
            bar.set_linewidth(2)
        for bar in errorbar_container4[2]:
            bar.set_linestyle(":")
            bar.set_color("r")
            bar.set_linewidth(2)

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="blue",
                linewidth=2,
                linestyle="-",
                label="Data (barrel)",
            ),
            Line2D(
                [0], [0], color="red", linewidth=2, linestyle="-", label="Data (endcap)"
            ),
            Line2D(
                [0],
                [0],
                color="blue",
                linewidth=2,
                linestyle=":",
                label="Data - WZ (barrel)",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=2,
                linestyle=":",
                label="Data - WZ (endcap)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            handlelength=3,
            handleheight=2,
            fontsize=15,
            ncols=2,
        )
        hep.cms.text("Preliminary", ax=ax)
        hep.cms.lumitext(
            f"{self.luminosities[self.year] * 1e-3:.1f} fb$^{{-1}}$ ({self.year}, 13.6 TeV)",
            ax=ax,
        )
        xlabel = "$p_T(e)$ [GeV]" if category == "electron" else "$p_T(\mu)$ [GeV]"
        ax.set(ylim=ylim, xlabel=xlabel)
        ax.set_ylabel("Fake rate")
        output_path = Path(f"{self.output_dir}/{category}")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"{output_path}/{self.workflow}_{category}_FR_{self.year}.{extension}"
        )