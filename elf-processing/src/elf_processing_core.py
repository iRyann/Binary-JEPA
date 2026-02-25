import angr

"""
This module is aimed at providing all necessary stuff 
for performing our analysis.
It consists of the following parts:
- Lifting 
- CFG construction
"""


def lift_binary(binary_path):
    """
    Lifts the binary into an angr project.
    :param binary_path: path to the binary to be lifted
    :return: an angr project
    """
    project = angr.Project(binary_path, auto_load_libs=False)
    return project


def construct_cfg(project: angr.Project):
    """
    Constructs the control flow graph (CFG) of the given project.
    :param project: an angr project
    :return: a CFG object
    """

    cfg = project.analyses.CFGFast()
    return cfg
