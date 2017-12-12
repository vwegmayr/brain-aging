"""Setup script for ml-project"""
import sys
import subprocess
from os.path import normpath
import argparse


def setup(args):
    """Setup function

    Requires:
        Installation of `miniconda`_.

    Todo:
        * Include automatic installtion of miniconda.

    .. _miniconda:
       https://conda.io/docs/install/quick.html#linux-miniconda-install

    """

    PROJECT_NAME = "entrack"

    if sys.version_info.major < 3:
        action = getattr(subprocess, "call")
    elif sys.version_info.minor < 5:
        action = getattr(subprocess, "call")
    else:
        action = getattr(subprocess, "run")

    if args.conda or args.all:
        action(["conda", "env", "create", "-n", PROJECT_NAME, "-f",
                ".environment"])

    if args.smt or args.all:
        action(["bash", "-c", "source activate {} && ".format(PROJECT_NAME) +
                "smt init -d {datapath} -i {datapath} -e python -m run.py "
                "-c error -l cmdline {project_name}".format(
                datapath=normpath('./data'), project_name=PROJECT_NAME)])

    if args.config or args.all:
        action(["cp", "configs/example_config.yaml", "configs/config.yaml"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Setup of conda and sumatra.")

    parser.add_argument("--smt", action="store_true")
    parser.add_argument("--conda", action="store_true")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    setup(args)
