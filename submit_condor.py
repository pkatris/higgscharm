import os
import json
import argparse
import subprocess
from pathlib import Path
from analysis.utils import make_output_directory
from analysis.filesets.utils import divide_list, get_nano_version, fileset_checker


def move_proxy() -> str:
    """move x509 proxy file from /tmp to /afs/private. Returns the afs path"""
    try:
        subprocess.run("voms-proxy-info -exists -valid 0:20", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expired or non-existing: please run 'voms-proxy-init --voms cms'"
        )
    user = os.environ["USER"]
    x509_localpath = subprocess.check_output(
        "voms-proxy-info -path", shell=True, text=True
    ).strip()
    x509_path = (
        f"/afs/cern.ch/user/{user[0]}/{user}/private/{x509_localpath.split('/')[-1]}"
    )
    subprocess.run(["cp", x509_localpath, x509_path])
    return x509_path


if __name__ == "__main__":
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
        dest="year",
        type=str,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
            "2024",
        ],
        help="dataset year",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=10,
        help="number of root files to include in each dataset partition (default 10)",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Enable Condor job submission. If not provided, it just builds condor files",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "root", "parquet"],
        help="format of output histogram",
    )
    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        default="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.30-py3.10",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="20000",
        help="Requested memory (in MB) for the condor job",
    )
    parser.add_argument("--jobflavor", dest="jobflavor", type=str, default="longlunch")
    args = parser.parse_args()

    # check if the fileset for the given year exists, generate it otherwise
    fileset_checker(samples=[args.dataset], year=args.year)

    print(f"Creating {args.workflow}-{args.year}-{args.dataset} condor file")
    jobname = f"{args.workflow}_{args.dataset}"

    # make condor and log directories
    condor_dir = Path.cwd() / "condor"
    job_dir = condor_dir / args.workflow / args.year / args.dataset
    if not job_dir.exists():
        job_dir.mkdir(parents=True, exist_ok=True)

    log_dir = condor_dir / "logs" / args.workflow / args.year / args.dataset
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # save partitions json and jobnums to job directory
    jobnum_list = []
    partition_dataset = {}
    fileset_path = Path.cwd() / "analysis" / "filesets"

    nano_version = get_nano_version(args.year)
    with open(
        f"{fileset_path}/fileset_{args.year}_nanov{nano_version}_lxplus.json", "r"
    ) as f:
        root_files = json.load(f)[args.dataset]
    root_files_list = divide_list(root_files, args.nfiles)
    for i in range(len(root_files_list)):
        dataset_key = (
            f"{args.dataset}_{i+1}" if len(root_files_list) > 1 else args.dataset
        )
        partition_dataset[i + 1] = {dataset_key: root_files_list[i]}
        jobnum_list.append(i + 1)

    partition_file = job_dir / "partitions.json"
    with open(f"{partition_file}", "w") as json_file:
        json.dump(partition_dataset, json_file, indent=4)

    jobnum_file = job_dir / "jobnum.txt"
    with open(f"{jobnum_file}", "w") as f:
        print(*jobnum_list, sep="\n", file=f)

    # build and save arguments json
    args.output_path = make_output_directory(args)
    args.user = os.environ["USER"]
    args_file = job_dir / "arguments.json"
    with open(args_file, "w") as json_file:
        json.dump(vars(args), json_file, indent=4)

    # make condor file
    local_condor = f"{job_dir}/{jobname}.sub"
    with open(f"{condor_dir}/submit.sub") as condor_template_file, open(
        local_condor, "w"
    ) as condor_file:
        for line in condor_template_file:
            line = line.replace("CONDORDIR", str(condor_dir))
            line = line.replace("BASEDIR", str(Path.cwd()))
            line = line.replace("X509PATH", move_proxy())
            line = line.replace("LOGDIR", str(log_dir))
            line = line.replace("JOBNAME", jobname)
            line = line.replace(
                "INPUTFILES", f"{partition_file},{jobnum_file},{args_file}"
            )
            line = line.replace("JOBNUM_FILE", str(jobnum_file))
            line = line.replace("JOBFLAVOR", f'"{args.jobflavor}"')
            line = line.replace("IMAGE", f'"{args.image}"')
            line = line.replace("MEMORY", args.memory)
            condor_file.write(line)

    if args.submit:
        subprocess.run(["condor_submit", local_condor])
