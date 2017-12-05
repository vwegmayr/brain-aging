"""Setup script for ml-project"""
import sys
import subprocess
from os.path import normpath


def setup():
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

    action(["conda", "env", "create", "-n", PROJECT_NAME, "-f",
            ".environment"])

    action(["bash", "-c", "source activate {} && ".format(PROJECT_NAME) +
            "smt init -d {datapath} -i {datapath} -e python -m run.py "
            "-c error -l cmdline {project_name}".format(
            datapath=normpath('./data'), project_name=PROJECT_NAME)])

    action(["cp", ".example_config.yaml", ".config.yaml"])

    print("\n========================================================")
    print("Type 'source activate {}' ".format(PROJECT_NAME) +
          "to activate environment.")
    print("==========================================================")


if __name__ == '__main__':
    setup()
