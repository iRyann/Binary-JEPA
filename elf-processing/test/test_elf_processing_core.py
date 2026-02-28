import os

import src.angr_utils as au
import src.elf_processing_core as elc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_hello_world():
    binary_path = os.path.join(BASE_DIR, "external", "HelloWorld", "hello_world")


if __name__ == "__main__":
    test_hello_world()
