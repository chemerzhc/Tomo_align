import argparse
import os
import sys
from tomography.utils.config import load_config
from tomography.utils.logger import setup_logger

############RUNING COMMAND#########################
# python run.py --params_path tomography/params/default.yml
# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--params_path",
    type=str,
    required=True,
    help="Path to yaml config"
)

args = parser.parse_args()

config = load_config(args.params_path)

# =========================================================
# Project root
# =========================================================
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# logger
# =========================================================
output_dir = config["project"]["output_dir"]

logger = setup_logger(output_dir)

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Using config: {args.params_path}")

# =========================================================
# GPU / CPU backend
# =========================================================
use_gpu = config["gpu"]["use_gpu"]

if use_gpu:
    try:
        import cupy as cp
        GPU_AVAILABLE = True
        logger.info("CuPy detected -> GPU enabled")

    except ImportError:
        import numpy as cp
        GPU_AVAILABLE = False

        logger.warning(
            "CuPy unavailable -> CPU mode"
        )

else:
    import numpy as cp
    GPU_AVAILABLE = False
    logger.info("CPU mode forced")

# =========================================================
# Imports
# =========================================================
from tomography.preprocess.pipeline import preprocess_emd_folder

from tomography.alignment.align_pipline import (
    run_alignment_pipeline
)

from tomography.alignment.common_line_align_pipline import (
    com_align_pip
)

from tomography.alignment.fine_align_pipline import (
    run_pc_refinement_pipeline
)

# =========================================================
# Step control
# =========================================================
steps = config["steps"]

# =========================================================
# Stage 0 preprocess
# =========================================================
if steps["preprocess"]:

    logger.info("Running preprocess...")

    p = config["preprocess"]

    preprocess_emd_folder(
        emd_dir=p["emd_dir"],
        output_h5=p["output_h5"]
    )

    logger.info("Preprocess finished.")

# =========================================================
# Stage 1 global alignment
# =========================================================
if steps["global_align"]:

    logger.info(
        "Running global alignment..."
    )

    p = config["global_alignment"]

    run_alignment_pipeline(**p)

    logger.info(
        "Global alignment finished."
    )

# =========================================================
# Stage 2 common line
# =========================================================
if steps["common_line"]:

    logger.info(
        "Running common-line alignment..."
    )

    p = config["common_line"]

    com_align_pip(**p)

    logger.info(
        "Common-line alignment finished."
    )

# =========================================================
# Stage 3 PC refinement
# =========================================================
if steps["pc_refine"]:

    logger.info(
        "Running PC refinement..."
    )

    p = config["pc_refinement"]

    run_pc_refinement_pipeline(**p)

    logger.info(
        "PC refinement finished."
    )

logger.info("Pipeline completed.")