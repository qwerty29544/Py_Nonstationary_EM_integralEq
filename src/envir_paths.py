import os

PATH_TO_PROJECT = os.path.dirname(os.getcwd())
FIGURES_PATH = os.path.join(PATH_TO_PROJECT, "figures")
COEFFS_PATH = os.path.join(PATH_TO_PROJECT, "coeffs")
LOGS_PATH = os.path.join(PATH_TO_PROJECT, "logs")
GRIDS_PATH = os.path.join(PATH_TO_PROJECT, "grids")
SRC_PATH = os.path.join(PATH_TO_PROJECT, "src")
TEST_PATH = os.path.join(PATH_TO_PROJECT, "test")


def create_base_proj_structure():
    if os.path.exists(FIGURES_PATH) is False:
        os.mkdir(FIGURES_PATH)
    if os.path.exists(COEFFS_PATH) is False:
        os.mkdir(COEFFS_PATH)
    if os.path.exists(LOGS_PATH) is False:
        os.mkdir(LOGS_PATH)
    if os.path.exists(GRIDS_PATH) is False:
        os.mkdir(GRIDS_PATH)