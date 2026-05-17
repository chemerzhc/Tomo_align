import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import astra

from scipy.ndimage import zoom
from typing import List, Tuple


def load_tilt_series(
    aligned_h5_file: str,
    downsample_ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load tilt-series data and optionally apply spatial downsampling.

    Args:
        aligned_h5_file:
            Path to aligned HDF5 file containing
            ``tiltSeries`` and ``tiltAngles``.

        downsample_ratio:
            Spatial downsampling factor.

    Returns:
        Downsampled tilt-series and tilt angles.
    """
    with h5py.File(aligned_h5_file, "r") as f:
        tilt_series = f["tiltSeries"][:].astype(np.float32)
        tilt_angles = f["tiltAngles"][:].astype(np.float32)

    tilt_series_ds = zoom(
        tilt_series,
        (downsample_ratio, downsample_ratio, 1),
        order=1
    )

    return tilt_series_ds, tilt_angles


def prepare_astra_geometry(
    tilt_series: np.ndarray,
    tilt_angles: np.ndarray
):
    """
    Create ASTRA projection and volume geometry.

    Args:
        tilt_series:
            Tilt-series array of shape
            ``(Nz, Ndet, Nproj)``.

        tilt_angles:
            Tilt angles in degrees.

    Returns:
        Projection geometry, volume geometry,
        and ASTRA-compatible projections.
    """
    nz, ndet, _ = tilt_series.shape

    projections_3d = np.transpose(
        tilt_series, (0, 2, 1)
    )

    projections_3d = np.ascontiguousarray(
        projections_3d
    )

    angles_rad = np.deg2rad(tilt_angles)

    proj_geom = astra.create_proj_geom(
        "parallel3d",
        1.0,
        1.0,
        nz,
        ndet,
        angles_rad
    )

    vol_geom = astra.create_vol_geom(
        ndet, ndet, nz
    )

    return proj_geom, vol_geom, projections_3d


def run_sirt3d(
    projections_3d: np.ndarray,
    proj_geom,
    vol_geom,
    total_iter: int = 2000,
    check_interval: int = 10
):
    """
    Run GPU-accelerated 3D SIRT reconstruction
    with residual monitoring.

    Args:
        projections_3d:
            ASTRA-compatible projections.

        proj_geom:
            ASTRA projection geometry.

        vol_geom:
            ASTRA volume geometry.

        total_iter:
            Total SIRT iterations.

        check_interval:
            Monitoring interval.

    Returns:
        Reconstructed volume, residual history,
        and intermediate slices.
    """
    print("[SIRT3D] Initializing SIRT3D_CUDA...")

    projections_id = astra.data3d.create(
        "-proj3d",
        proj_geom,
        projections_3d
    )

    reconstruction_id = astra.data3d.create(
        "-vol",
        vol_geom,
        0
    )

    cfg = astra.astra_dict("SIRT3D_CUDA")
    cfg["ProjectionDataId"] = projections_id
    cfg["ReconstructionDataId"] = reconstruction_id
    cfg["option"] = {"MinConstraint": 0.0}

    alg_id = astra.algorithm.create(cfg)

    residuals: List[float] = []
    evolution_slices = []

    mid_z = projections_3d.shape[0] // 2

    print(f"{'Iteration':<15}| {'Residual':<20}")
    print("-" * 40)

    for i in range(0, total_iter, check_interval):
        astra.algorithm.run(
            alg_id,
            check_interval
        )

        residual = astra.algorithm.get_res_norm(
            alg_id
        )

        residuals.append(residual)

        current_vol = astra.data3d.get(
            reconstruction_id
        )

        evolution_slices.append(
            current_vol[:, :, mid_z].copy()
        )

        print(
            f"{i + check_interval:<15}"
            f"| {residual:<20.6e}"
        )

    volume_out = astra.data3d.get(
        reconstruction_id
    )

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(reconstruction_id)

    return (
        volume_out,
        residuals,
        evolution_slices
    )