import os

import src.elf_processing_core as elc
import src.angr_utils as au

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_hello_world():
    binary_path = os.path.join(BASE_DIR, "external", "HelloWorld", "hello_world")

    project = elc.lift_binary(binary_path)
    cfg = elc.construct_cfg(project)
    au.print_cfg(cfg, project)

if __name__ == "__main__":
    test_hello_world()
