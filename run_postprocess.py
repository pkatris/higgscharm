import gc
import yaml
import json
import glob
import logging
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.postprocess.plotter import CoffeaPlotter
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.filesets.utils import get_workflow_key_process_map, get_process_sample_map
from analysis.postprocess.postprocessor import (
    save_histograms_by_sample,
    save_histograms_by_process,
)
from analysis.postprocess.utils import (
    print_header,
    setup_logger,
    clear_output_directory,
    df_to_latex,
    combine_event_tables,
    combine_cutflows,
    format_cutflow_with_efficiency,
    merge_parquets_by_sample,
    load_processed_histograms,
    get_results_report,
)


OUTPUT_DIR = Path("/eos/user/p/pkatris/higgscharm/outputs")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workflow",
        dest="workflow",
        required=True,
        type=str,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="workflow to run",
    )
    parser.add_argument(
        "-y",
        "--year",
        required=True,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2016",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2022",
            "2023preBPix",
            "2023postBPix",
            "2023",
            "2024",
        ],
        help="Data year",
    )
    parser.add_argument(
        "--log", action="store_true", help="Enable log scale for y-axis"
    )
    parser.add_argument(
        "--postprocess", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--yratio_limits",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
        help="Set y-axis ratio limits",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output file extension for plots",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="process",
        help="Axis to group by (e.g., 'process', or a JSON dict)",
    )
    parser.add_argument(
        "--pass_axis",
        type=str,
        default=None,
        help="Binary axis (e.g., 'is_passing_lepton')",
    )
    parser.add_argument(
        "--nocutflow", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="Format of output files",
    )
    parser.add_argument(
        "--skipmerging", action="store_true", help="Skip parquet outputs merging"
    )
    parser.add_argument("--blind", action="store_true", help="Blind data")
    parser.add_argument(
        "--blind_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Blind data in [LOW, HIGH] of the variables in --blind_vars (e.g. 100 150)",
    )
    parser.add_argument(
        "--blind_vars",
        type=str,
        nargs="+",
        default=None,
        help="Variables to apply --blind_range to (default: all plotted variables)",
    )
    parser.add_argument(
        "--rebin",
        type=str,
        nargs="+",
        default=None,
        metavar="VAR=SPEC",
        help=(
            "Rebin at plotting time. SPEC is an integer factor (e.g. zz_mass_inclusive=3 "
            "turns 5 GeV into 15 GeV bins) or comma-separated new edges that are a subset "
            "of the original ones (e.g. zz_mass_inclusive=70,80,85,90,95,100,105,...,400). "
            "Use 'zz_mass_*=3' to apply to every variable matching the pattern."
        ),
    )
    return parser.parse_args()


def parse_rebin(rebin_args, variables):
    """['var=3', 'zz_mass_*=70,80,...'] -> {variable: 3 or [edges]}"""
    import fnmatch

    rebin = {}
    for item in rebin_args or []:
        if "=" not in item:
            raise ValueError(f"--rebin expects VAR=SPEC, got '{item}'")
        pattern, spec = item.split("=", 1)
        value = (
            [float(x) for x in spec.split(",") if x.strip()] if "," in spec else int(spec)
        )
        matched = [v for v in variables if fnmatch.fnmatch(v, pattern)]
        if not matched:
            raise ValueError(f"--rebin: no histogram variable matches '{pattern}'")
        for v in matched:
            rebin[v] = value
    return rebin


def check_output_dir(workflow: str, year: str) -> Path:
    """
    Verify that the output directory exists for the given workflow and year.
    - For years 2022 and 2023, both pre/post sub-year directories must exist
      before creating the parent directory.
    - Returns the valid Path if successful.
    - Raises FileNotFoundError if required directories are missing.
    """

    output_dir = OUTPUT_DIR / workflow / year

    if output_dir.exists():
        return output_dir

    # Years that require both pre and post subdirectories
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }

    if year in aux_map:
        pre_year, post_year = [OUTPUT_DIR / workflow / y for y in aux_map[year]]

        # Collect missing subdirectories
        missing = [str(p) for p in (pre_year, post_year) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required directories for year {year}: {', '.join(missing)}"
            )

        # Create the parent directory if both pre and post exist
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Case for sub-years or any other invalid year
    raise FileNotFoundError(f"Could not find outputs at {output_dir}")


def get_sample_name(filename: str, year: str) -> str:
    """return sample name from filename"""
    sample_name = Path(filename).stem
    if sample_name.rsplit("_")[-1].isdigit():
        sample_name = "_".join(sample_name.rsplit("_")[:-1])
    return sample_name.replace(f"{year}_", "")


def load_year_histograms(workflow: str, year: str):
    """load and merge histograms from pre/post years"""
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }
    pre_year, post_year = aux_map[year]
    base_path = OUTPUT_DIR / workflow
    pre_file = base_path / pre_year / f"{pre_year}_processed_histograms.coffea"
    post_file = base_path / post_year / f"{post_year}_processed_histograms.coffea"
    return accumulate([load(pre_file), load(post_file)])


def load_histogram_file(path: Path):
    return load(path) if path.exists() else None


def plot_variable(variable: str, group_by, histogram_config) -> bool:
    """decide whether to plot a given variable under group_by mode"""
    if isinstance(group_by, str) and group_by == "process":
        return True
    for hist_key, variables in histogram_config.layout.items():
        if variable in variables and group_by["name"] in variables:
            return group_by["name"] != variable
    return False


def get_blind_range(variable: str, blind_range, blind_vars):
    """return the blinding window for this variable, or None if it isn't blinded"""
    if blind_range is None:
        return None
    if blind_vars is None or variable in blind_vars:
        return tuple(blind_range)
    return None


if __name__ == "__main__":
    args = parse_arguments()

    try:
        group_by = json.loads(args.group_by)
    except json.JSONDecodeError:
        group_by = args.group_by

    output_dir = check_output_dir(args.workflow, args.year)
    clear_output_directory(output_dir, "txt")
    setup_logger(output_dir)

    config_builder = WorkflowConfigBuilder(workflow=args.workflow)
    workflow_config = config_builder.build_workflow_config()
    histogram_config = workflow_config.histogram_config
    event_selection = workflow_config.event_selection
    categories = event_selection["categories"]
    processed_histograms = None

    if "data" not in workflow_config.datasets:
        args.blind = True

    if args.blind_range is not None:
        low, high = args.blind_range
        if low >= high:
            raise ValueError(
                f"--blind_range LOW must be smaller than HIGH (got {low}, {high})"
            )

    if args.postprocess and (args.year not in ["2016", "2022", "2023"]):
        print_header(f"Running postprocess for {args.year}")
        logging.info(f"Reading outputs from: {output_dir}")

        output_files = [
            f
            for f in glob.glob(f"{output_dir}/*/*.coffea", recursive=True)
            if not Path(f).stem.startswith("cutflow")
            and not Path(f).stem.startswith("processed")
        ]

        grouped_outputs = defaultdict(list)
        for output_file in output_files:
            sample_name = get_sample_name(output_file, args.year)
            grouped_outputs[sample_name].append(output_file)

        process_samples_map = get_process_sample_map(grouped_outputs.keys(), args.year)

        if args.output_format == "parquet":
            if not args.skipmerging:
                merge_parquets_by_sample(output_dir, args.year, categories)

        for sample in grouped_outputs:
            save_histograms_by_sample(
                grouped_outputs,
                sample,
                args.year,
                output_dir,
                categories,
                workflow_config,
                args.nocutflow,
                args.output_format,
                args.skipmerging,
            )
            gc.collect()

        for process in process_samples_map:
            save_histograms_by_process(
                process,
                output_dir,
                process_samples_map,
                categories,
                args.nocutflow,
                args.output_format,
            )
            gc.collect()

        processed_histograms = load_processed_histograms(
            args.year,
            output_dir,
            process_samples_map,
        )

        for category in categories:
            logging.info(f"category: {category}")
            category_dir = output_dir / str(category)

            if not args.nocutflow:
                print_header(f"Cutflow")
                cutflow_df = pd.DataFrame()
                for process in process_samples_map:
                    cutflow_file = category_dir / f"cutflow_{category}_{process}.csv"
                    cutflow_df = pd.concat(
                        [cutflow_df, pd.read_csv(cutflow_file, index_col=[0])], axis=1
                    )

                columns_to_drop = []
                key_process_map = get_workflow_key_process_map(
                    workflow_config, args.year
                )
                if "signal" in workflow_config.datasets:
                    signal_keys = [k for k in workflow_config.datasets["signal"]]
                    signals = [key_process_map[key] for key in signal_keys]
                    columns_to_drop += signals

                if not args.blind:
                    columns_to_drop += ["Data"]

                total_background = cutflow_df.drop(columns=columns_to_drop).sum(axis=1)
                cutflow_df["Total Background"] = total_background

                cutflow_index = event_selection["categories"][category]
                cutflow_df = cutflow_df.loc[cutflow_index]

                if not args.blind:
                    to_process = ["Data", "Total Background"]
                else:
                    to_process = ["Total Background"]
                cutflow_df = cutflow_df[
                    to_process
                    + [
                        process
                        for process in cutflow_df.columns
                        if process not in to_process
                    ]
                ]
                logging.info(
                    f'{cutflow_df.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")}\n'
                )
                cutflow_df.to_csv(f"{category_dir}/cutflow_{category}.csv")
                logging.info("\n")

            if args.workflow in ["ztoee", "ztomumu"]:
                print_header(f"Results")
                results_df = get_results_report(
                    processed_histograms,
                    workflow_config,
                    category,
                    columns_to_drop,
                    args.blind,
                )
                logging.info(
                    results_df.applymap(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
                )
                logging.info("\n")
                results_df.to_csv(f"{category_dir}/results_{category}.csv")

                # save latex table
                latex_table = df_to_latex(results_df, args.blind)
                with open(category_dir / f"results_{category}.txt", "w") as f:
                    f.write(latex_table)

    if args.year in ["2016", "2022", "2023"]:
        if args.postprocess:
            print_header(f"Running postprocess for {args.year}")
            # load and accumulate processed histograms
            processed_histograms = load_year_histograms(args.workflow, args.year)
            save(
                processed_histograms,
                f"{output_dir}/{args.year}_processed_histograms.coffea",
            )
            identifier_map = {"2016": "VFP", "2022": "EE", "2023": "BPix"}
            identifier = identifier_map[args.year]
            for category in categories:
                logging.info(f"category: {category}")
                category_dir = OUTPUT_DIR / args.workflow / args.year / category
                if not category_dir.exists():
                    category_dir.mkdir(parents=True, exist_ok=True)
                if args.workflow in ["ztoee", "ztomumu"]:
                    # load and combine results tables
                    results_pre = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}pre{identifier}"
                        / category
                        / f"results_{category}.csv",
                        index_col=0,
                    )
                    results_post = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}post{identifier}"
                        / category
                        / f"results_{category}.csv",
                        index_col=0,
                    )
                    combined_results = combine_event_tables(
                        results_pre, results_post, args.blind
                    )

                    print_header(f"Results")
                    logging.info(
                        combined_results.applymap(
                            lambda x: f"{x:.5f}" if pd.notnull(x) else ""
                        )
                    )
                    logging.info("\n")

                    combined_results.to_csv(category_dir / f"results_{category}.csv")

                    # save latex table
                    latex_table = df_to_latex(combined_results, args.blind)
                    with open(category_dir / f"results_{category}.txt", "w") as f:
                        f.write(latex_table)

                # load and combine cutflow tables
                if not args.nocutflow:
                    print_header(f"Cutflow")
                    cutflow_pre = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}pre{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    cutflow_post = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}post{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    combined_cutflow = combine_cutflows(cutflow_pre, cutflow_post)
                    combined_cutflow.to_csv(category_dir / f"cutflow_{category}.csv")
                    logging.info(
                        combined_cutflow.applymap(
                            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
                        )
                    )
                    logging.info("\n")

                    # compute efficiencies
                    print_header(f"Efficiency")
                    eff_df = pd.DataFrame(index=combined_cutflow.index)
                    for col in combined_cutflow.columns:
                        eff_df[col] = (
                            combined_cutflow[col] / combined_cutflow[col].iloc[0] * 100
                        )
                    eff_df.to_csv(category_dir / f"eff_{category}.csv")
                    logging.info(eff_df)
                    logging.info("\n")

                    cutflow_eff = format_cutflow_with_efficiency(
                        combined_cutflow, eff_df
                    )
                    cutflow_eff.to_csv(category_dir / f"cutflow_eff_{category}.csv")

    if args.plot:
        subprocess.run("python3 analysis/postprocess/build_color_map.py", shell=True)
        if not args.postprocess:
            postprocess_file = output_dir / f"{args.year}_processed_histograms.coffea"
            processed_histograms = load_histogram_file(postprocess_file)
            if processed_histograms is None:
                raise ValueError(
                    f"Postprocess file not found. Please run:\n"
                    f"  'python3 run_postprocess.py -w {args.workflow} -y {args.year} --postprocess'"
                )

        print_header(f"Running plotter for {args.year}")
        plotter = CoffeaPlotter(
            workflow=args.workflow,
            processed_histograms=processed_histograms,
            year=args.year,
            output_dir=output_dir,
            group_by=group_by,
            pass_axis=args.pass_axis,
            rebin=parse_rebin(args.rebin, list(histogram_config.axes.keys())),
        )
        for category in categories:
            logging.info(
                f"Plotting histograms by '{group_by if group_by == 'process' else group_by['name']}' for category '{category}'"
            )
            for variable in workflow_config.histogram_config.variables:
                if args.pass_axis:
                    if variable == args.pass_axis:
                        continue
                    if histogram_config.layout == "individual":
                        print("There's only individual axes!")
                        break
                    proceed = False
                    for key, variables in histogram_config.layout.items():
                        if (variable in variables) and (args.pass_axis in variables):
                            proceed = True
                            break
                    if not proceed:
                        continue
                if plot_variable(variable, group_by, workflow_config.histogram_config):
                    logging.info(variable)
                    blind_range = get_blind_range(
                        variable, args.blind_range, args.blind_vars
                    )
                    if blind_range is not None:
                        logging.info(
                            f"  blinding data in [{blind_range[0]}, {blind_range[1]}]"
                        )
                    plotter.plot_histograms(
                        variable=variable,
                        category=category,
                        yratio_limits=args.yratio_limits,
                        log=args.log,
                        extension=args.extension,
                        blind=args.blind,
                        blind_range=blind_range,
                    )
            if args.workflow.startswith("zplusl_"):
                plotter.plot_fake_rate(category)
            subprocess.run(
                f"tar -zcvf {output_dir}/{category}/{args.workflow}_{args.year}_plots.tar.gz {output_dir}/{category}/*.{args.extension}",
                shell=True,
            )

        # Save fake rates once per year, computed directly from this plotter's
        # already-loaded Data/WZ histograms (reusing the same barrel/endcap
        # DATA-minus-WZ logic as plot_fake_rate/_debug_fr_table above),
        # instead of relying on the separate, currently broken
        # compute_fake_rates.py script.
        #
        # Written to the same path estimate_zx_background.py expects:
        #   OUTPUT_DIR/fake_rates/{year}/fake_rates.pkl
        #
        # NOTE: only zplusl_ss and zplusl_os carry a "WZ" process in
        # processed_histograms (see _is_wz_dataset filtering upstream in
        # compute_fake_rates.py / load_zplusl_histograms). If you also run
        # zplusl_ss_maximal/_minimal/_intermediate variants through --plot
        # and they do NOT have a "WZ" key, narrow this condition to
        # `args.workflow in ("zplusl_ss", "zplusl_os")` to avoid a KeyError.
        if args.workflow.startswith("zplusl_"):
            print_header(f"Saving fake rates for {args.year}")
            fake_rates_path = OUTPUT_DIR / "fake_rates" / args.year / "fake_rates.pkl"
            plotter.save_fake_rates(
                output_path=fake_rates_path,
                categories=tuple(categories),
            )