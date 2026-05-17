from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from tomography.preprocess.emd_reader import load_emd_frames_and_angle
from tomography.preprocess.frame_align import (
    align_and_sum_frames,
    visualize_shifts,
)
from tomography.preprocess.normalize import robust_normalize
from tomography.preprocess.hdf5_writer import write_tomo_hdf5


def preprocess_emd_folder(
    emd_dir,
    output_h5,
    n_frames=4,
    normalize=True,
    norm_percentile=99.9,
    DCF_debug=True,
):
    """
    Preprocess a folder of EMD tomography files.

    Workflow
    --------
    - Load EMD files
    - Align and sum frames for each file
    - Optionally visualize shifts for debugging
    - Optionally normalize projections
    - Sort projections by tilt angle
    - Save processed data to HDF5

    Parameters
    ----------
    emd_dir : str or Path
        Directory containing EMD files.
    output_h5 : str or Path
        Output HDF5 file path.
    n_frames : int, optional
        Expected number of frames per EMD file.
    normalize : bool, optional
        Whether to apply robust normalization.
    norm_percentile : float, optional
        Percentile used for robust normalization.
    DCF_debug : bool, optional
        Whether to visualize shifts for the first three files.

    Returns
    -------
    None
    """
    emd_dir = Path(emd_dir)
    emd_files = list(emd_dir.glob("*.emd"))

    angles = []
    projections = []

    for idx, f in enumerate(
        tqdm(emd_files, desc="Processing EMDs")
    ):
        frames, angle_array = load_emd_frames_and_angle(f)

        angle = angle_array[0]
        angle_mean = np.mean(angle_array)

        print(f"\nFile {idx+1}/{len(emd_files)}: {f.name}")
        print("  Original frames shape:", frames.shape)
        print("  Tilt angle (taken first):", angle)
        print("  Tilt angle mean:", angle_mean)

        if frames.shape[0] != n_frames:
            raise ValueError(
                f"{f} has {frames.shape[0]} frames"
            )

        summed, shifts_list = align_and_sum_frames(frames)

        print(
            "  Summed shape after align_and_sum_frames:",
            summed.shape
        )

        if DCF_debug and idx < 3:
            visualize_shifts(shifts_list)

        if normalize:
            summed = robust_normalize(
                summed,
                norm_percentile
            )
            print(
                "  Summed shape after normalization:",
                summed.shape
            )

        projections.append(summed)
        angles.append(angle)

        print(
            "  Current projections list length:",
            len(projections)
        )

    angles = np.asarray(angles)
    projections = np.stack(projections, axis=0)

    print("Final angles shape:", angles.shape)
    print(
        "Final projections shape:",
        projections.shape
    )

    order = np.argsort(angles)
    angles = angles[order]
    projections = projections[order]

    write_tomo_hdf5(
        output_h5,
        angles,
        projections,
        attrs={
            "normalization":
                f"robust_p{norm_percentile}",
            "n_frames": n_frames,
            "detector": "HAADF",
            "source": "hyperspy emd",
        },
    )