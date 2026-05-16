# -------------------------------------------------------------------------------------------------
# ASTRA SIRT reconstruction
# -------------------------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt


def save_theta0_simple_projection(
    volume,
    out_dir="./output",
    fname="proj_theta0.png"
):
    """
    Simple sanity-check projection at tilt angle = 0 deg.

    Assumes volume shape = (Nx, Ny, Nz)
    Projection = sum over z
    """

    os.makedirs(out_dir, exist_ok=True)

    proj0 = volume.sum(axis=2)

    # Normalize for visualization only
    proj0 = proj0 - proj0.min()

    if proj0.max() > 0:
        proj0 = proj0 / proj0.max()

    out_path = os.path.join(out_dir, fname)

    plt.imsave(out_path, proj0, cmap="gray")

    print(
        f"[PC-check] θ=0 simple projection saved to: "
        f"{out_path}"
    )

    return proj0


import astra
from scipy.ndimage import zoom
import numpy as np
import astra
from scipy.ndimage import zoom


def astra_slice_reconstruction(
    tilt_series,
    tilt_angles,
    Nx_target=None,
    Ny_target=None,
    Nz_target=None,
    algorithm="SIRT",
    num_iter=100,
    use_gpu=True,
    verbose=True
):
    """
    ASTRA 3D reconstruction.

    Interface kept identical while reconstruction logic
    is upgraded to full 3D.

    Parameters
    ----------
    tilt_series : ndarray
        Shape = (Nx, Ny, Nproj)

    tilt_angles : ndarray
        Tilt angles in degrees, shape = (Nproj,)
    """

    Nx, Ny, Nproj = tilt_series.shape

    # ---------------------------
    # Downsampling
    # ---------------------------
    if Nx_target is not None and Ny_target is not None:

        zoom_x = Nx_target / Nx
        zoom_y = Ny_target / Ny

        # Keep approximate isotropic scaling
        zoom_z = (
            (Nz_target / Nx)
            if Nz_target is not None
            else zoom_x
        )

        tilt_series = zoom(
            tilt_series,
            (zoom_z, zoom_y, 1),
            order=1
        )

        Nx, Ny, _ = tilt_series.shape

        if verbose:
            print(
                f"[astraSIRT] 3D downsampled → "
                f"({Nx}, {Ny}, {Nproj})"
            )

    # ---------------------------
    # Prepare ASTRA 3D geometry
    # ---------------------------
    projections_3d = np.transpose(
        tilt_series,
        (0, 2, 1)
    )

    projections_3d = np.ascontiguousarray(
        projections_3d
    )

    angles_rad = np.deg2rad(tilt_angles)

    proj_geom = astra.create_proj_geom(
        'parallel3d',
        1.0,
        1.0,
        Nx,
        Ny,
        angles_rad
    )

    vol_geom = astra.create_vol_geom(
        Ny,
        Ny,
        Nx
    )

    # ---------------------------
    # Algorithm name mapping
    # ---------------------------
    if algorithm.upper() == "FBP":

        alg_name = (
            "FBP3D_CUDA"
            if use_gpu
            else "FBP3D"
        )

    else:

        alg_name = (
            f"{algorithm.upper()}3D_CUDA"
            if use_gpu
            else f"{algorithm.upper()}3D"
        )

    # ---------------------------
    # Initialize ASTRA objects
    # ---------------------------
    proj_id = astra.data3d.create(
        '-proj3d',
        proj_geom,
        projections_3d
    )

    vol_id = astra.data3d.create(
        '-vol',
        vol_geom,
        0
    )

    cfg = astra.astra_dict(alg_name)

    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id

    if algorithm.upper() != "FBP":
        cfg['option'] = {
            'MinConstraint': 0.0
        }

    alg_id = astra.algorithm.create(cfg)

    # ---------------------------
    # Iterative reconstruction
    # ---------------------------
    check_interval = max(
        1,
        num_iter // 10
    )

    if verbose:

        print(
            f"[astraSIRT] Starting 3D "
            f"{algorithm} reconstruction..."
        )

        print(
            f"{'Iteration':<12} | "
            f"{'Residual (L2)':<18}"
        )

        print("-" * 35)

    for i in range(0, num_iter, check_interval):

        actual_step = min(
            check_interval,
            num_iter - i
        )

        astra.algorithm.run(
            alg_id,
            actual_step
        )

        res = astra.algorithm.get_res_norm(
            alg_id
        )

        if verbose:
            print(
                f"{i + actual_step:<12} | "
                f"{res:<18.6e}"
            )

    # ---------------------------
    # Retrieve reconstruction
    # ---------------------------
    volume_astra = astra.data3d.get(vol_id)

    astra.algorithm.delete(alg_id)

    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    # ---------------------------
    # Convert output format
    # ---------------------------
    volume = np.transpose(
        volume_astra,
        (1, 0, 2)
    )

    if verbose:
        print(
            f"[astraSIRT] Reconstruction finished. "
            f"Output shape: {volume.shape}"
        )

    return volume


# -------------------------------------------------------------------------------------------------
# ASTRA Forward Projection
# -------------------------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import astra
import math

# --------------------------
# Forward projection
# --------------------------
import numpy as np
import astra


def forward_project_volume(
    volume_xyz,
    angles_rad,
    det_spacing=1.0
):
    """
    3D forward projection with built-in coordinate
    correction and optional mirror compensation.

    Parameters
    ----------
    volume_xyz : ndarray
        Shape = (Nx, Ny, Nz)

    angles_rad : ndarray
        Tilt angles in radians
    """

    # ---------------------------
    # Internal coordinate settings
    # ---------------------------
    ANGLE_OFFSET_DEG = 0

    FLIP_X = True
    FLIP_Y = False

    # ---------------------------
    # Volume preprocessing
    # ---------------------------
    vol_to_proj = volume_xyz.copy()

    if FLIP_X:
        vol_to_proj = vol_to_proj[::-1, :, :]

    if FLIP_Y:
        vol_to_proj = vol_to_proj[:, ::-1, :]

    Nx, Ny, Nz = vol_to_proj.shape

    # ---------------------------
    # Angle compensation
    # ---------------------------
    adjusted_angles = (
        angles_rad
        + np.deg2rad(ANGLE_OFFSET_DEG)
    )

    adjusted_angles = angles_rad

    # ---------------------------
    # Prepare ASTRA geometry
    # ---------------------------
    volume_astra = np.transpose(
        vol_to_proj,
        (1, 0, 2)
    )

    volume_astra = np.ascontiguousarray(
        volume_astra.astype(np.float32)
    )

    vol_geom = astra.create_vol_geom(
        Nx,
        Ny,
        Nz
    )

    proj_geom = astra.create_proj_geom(
        'parallel3d',
        det_spacing,
        det_spacing,
        Nz,
        Nx,
        adjusted_angles
    )

    # ---------------------------
    # Execute forward projection
    # ---------------------------
    vol_id = astra.data3d.create(
        '-vol',
        vol_geom,
        volume_astra
    )

    sino_id = astra.data3d.create(
        '-proj3d',
        proj_geom
    )

    cfg = astra.astra_dict('FP3D_CUDA')

    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = sino_id

    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id)

    reproj = astra.data3d.get(sino_id)

    # ---------------------------
    # Cleanup GPU resources
    # ---------------------------
    astra.algorithm.delete(alg_id)

    astra.data3d.delete(vol_id)
    astra.data3d.delete(sino_id)

    # ---------------------------
    # Reformat output
    # ---------------------------
    reproj = np.transpose(
        reproj,
        (0, 2, 1)
    )

    return reproj


# -------------------------------------------------------------------------------------------------
# Apply high-pass filter to residual
# -------------------------------------------------------------------------------------------------
from scipy.fft import fft2, ifft2, fftshift


def highpass_filter(img, cutoff=0.15):
    """
    Parameters
    ----------
    img : ndarray
        Shape = (Ny, Nproj)
    """

    F = fftshift(fft2(img))

    Ny, Nx = img.shape

    cy, cx = Ny // 2, Nx // 2

    r = int(min(Ny, Nx) * cutoff)

    Y, X = np.ogrid[:Ny, :Nx]

    mask = (
        (Y - cy) ** 2
        + (X - cx) ** 2
        > r ** 2
    )

    F *= mask

    return np.real(
        ifft2(fftshift(F))
    )


# -------------------------------------------------------------------------------------------------
# Estimate shift from residual
# -------------------------------------------------------------------------------------------------
def estimate_shift_from_residual(residual):
    """
    Parameters
    ----------
    residual : ndarray
        Shape = (Ny, Nproj)

    Returns
    -------
    shifts : ndarray
        Shape = (Nproj, 2)
    """

    ref = residual[:, residual.shape[1] // 2]

    shifts = []

    for i in range(residual.shape[1]):

        corr = np.fft.ifft2(
            np.fft.fft2(ref)
            * np.conj(
                np.fft.fft2(residual[:, i])
            )
        )

        dy, dx = np.unravel_index(
            np.argmax(np.abs(corr)),
            corr.shape
        )

        shifts.append([dy, dx])

    return np.array(shifts)


# -------------------------------------------------------------------------------------------------
# Fine reconstruction alignment iteration
# -------------------------------------------------------------------------------------------------
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy.fft import ifftshift


def fft_shift(img, dy, dx):
    """
    FFT-based subpixel shift for 2D images.

    Parameters
    ----------
    img : ndarray
        Shape = (Ny, Nx)

    dy, dx : float
        Subpixel shift values
    """

    Ny, Nx = img.shape

    ky = ifftshift(
        np.arange(-Ny // 2, Ny // 2)
    ) / Ny

    kx = ifftshift(
        np.arange(-Nx // 2, Nx // 2)
    ) / Nx

    KX, KY = np.meshgrid(kx, ky)

    phase = np.exp(
        -2j * np.pi * (
            KX * dx + KY * dy
        )
    )

    return np.real(
        ifft2(
            fft2(img) * phase
        )
    )


import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_pc_iteration(
    tilt_img,
    reproj_img,
    res_img,
    res_hp_img,
    shift_vec,
    iter_num,
    proj_idx
):
    """
    Save visualization for one projection-consistency iteration.
    """

    out_dir = os.path.join(
        os.getcwd(),
        "output"
    )

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes = axes.ravel()

    im0 = axes[0].imshow(
        tilt_img,
        cmap='gray'
    )

    axes[0].set_title("Tilt (aligned)")

    plt.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046
    )

    im1 = axes[1].imshow(
        reproj_img,
        cmap='gray'
    )

    axes[1].set_title("Reprojection")

    plt.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046
    )

    im2 = axes[2].imshow(
        res_img,
        cmap='seismic'
    )

    axes[2].set_title("Residual")

    plt.colorbar(
        im2,
        ax=axes[2],
        fraction=0.046
    )

    im3 = axes[3].imshow(
        res_hp_img,
        cmap='seismic'
    )

    axes[3].set_title(
        "Residual (High-pass)"
    )

    plt.colorbar(
        im3,
        ax=axes[3],
        fraction=0.046
    )

    axes[4].axis("off")
    axes[5].axis("off")

    fig.suptitle(
        f"PC Iter {iter_num + 1} | "
        f"Proj {proj_idx} | "
        f"shift = (dy={shift_vec[0]:.3f}, "
        f"dx={shift_vec[1]:.3f})",
        fontsize=14
    )

    save_path = os.path.join(
        out_dir,
        f"pc_iter_{iter_num:03d}_proj_{proj_idx:03d}.png"
    )

    fig.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def projection_consistency_refinement(
    tilt_series,
    reproj_series,
    max_iter=10,
    lr=0.25,
    hp_cutoff=0.15,
    tol=1e-3
):
    """
    Projection-consistency alignment
    in projection space only.
    """

    assert tilt_series.shape == reproj_series.shape, (
        f"shape mismatch: "
        f"{tilt_series.shape} vs "
        f"{reproj_series.shape}"
    )

    Ny, Nx, Nproj = tilt_series.shape

    shift_record = np.zeros(
        (Nproj, 2),
        dtype=np.float32
    )

    for it in range(max_iter):

        print(
            f"[PC-align] Iter "
            f"{it + 1}/{max_iter}"
        )

        total_res = 0.0

        for ip in range(Nproj):

            res = (
                tilt_series[:, :, ip]
                - reproj_series[:, :, ip]
            )

            res_hp = highpass_filter(
                res,
                cutoff=hp_cutoff
            )

            F1 = fft2(res_hp)

            F2 = fft2(
                reproj_series[:, :, ip]
            )

            corr = ifft2(
                F1 * np.conj(F2)
            )

            dy, dx = np.unravel_index(
                np.argmax(np.abs(corr)),
                corr.shape
            )

            if dy > Ny // 2:
                dy -= Ny

            if dx > Nx // 2:
                dx -= Nx

            dy_corr = -lr * dy
            dx_corr = -lr * dx

            tilt_series[:, :, ip] = fft_shift(
                tilt_series[:, :, ip],
                dy=dy_corr,
                dx=dx_corr
            )

            shift_record[ip] = [
                dy_corr,
                dx_corr
            ]

            total_res += np.linalg.norm(
                res_hp
            )

        print(
            f"    residual norm = "
            f"{total_res:.4e}"
        )

        # Automatic visualization
        mid = Nproj // 2

        visualize_pc_iteration(
            tilt_img=tilt_series[:, :, mid],

            reproj_img=reproj_series[:, :, mid],

            res_img=(
                tilt_series[:, :, mid]
                - reproj_series[:, :, mid]
            ),

            res_hp_img=highpass_filter(
                tilt_series[:, :, mid]
                - reproj_series[:, :, mid],
                cutoff=hp_cutoff
            ),

            shift_vec=shift_record[mid],

            iter_num=it,

            proj_idx=mid
        )

        if total_res < tol:
            print("    Converged")
            break

    return tilt_series


import os
import math
import numpy as np
import matplotlib.pyplot as plt


class ProjectionMontage:
    """
    Create and save montage visualizations
    for projection series.
    """

    def __init__(
        self,
        angles_rad,
        n_show=48,
        n_cols=8,
        cmap="bwr"
    ):
        """
        Parameters
        ----------
        angles_rad : ndarray
            Tilt angles in radians

        n_show : int
            Number of projections to display

        n_cols : int
            Number of montage columns

        cmap : str
            Visualization colormap
        """

        self.angles_rad = angles_rad
        self.n_show = n_show
        self.n_cols = n_cols
        self.cmap = cmap

    @staticmethod
    def ensure_dimension_match(
        series,
        target_v,
        target_u,
        verbose=True
    ):
        """
        Ensure dimensions match target size.

        Only downsampling is applied.
        Upsampling is intentionally avoided.

        Parameters
        ----------
        series : ndarray
            Shape = (Ny_v, Nx_u, Nproj)
        """

        curr_v, curr_u, n_proj = series.shape

        scale_v = min(
            1.0,
            target_v / curr_v
        )

        scale_u = min(
            1.0,
            target_u / curr_u
        )

        if scale_v < 1.0 or scale_u < 1.0:

            if verbose:
                print(
                    f"[astraSIRT] Downsampling series: "
                    f"({curr_v}, {curr_u}) -> "
                    f"({int(curr_v * scale_v)}, "
                    f"{int(curr_u * scale_u)})"
                )

            series = zoom(
                series,
                (scale_v, scale_u, 1),
                order=1
            )

        return series

    @staticmethod
    def normalize_per_projection(
        series,
        eps=1e-8
    ):
        """
        Normalize each projection independently
        to the range [0, 1].
        """

        Ny, Nx, Nproj = series.shape

        series_norm = np.zeros_like(
            series,
            dtype=np.float32
        )

        for i in range(Nproj):

            s = series[:, :, i]

            s_min, s_max = (
                s.min(),
                s.max()
            )

            series_norm[:, :, i] = (
                (s - s_min)
                / (s_max - s_min + eps)
            )

        return series_norm

    def save_montage(
        self,
        series,
        out_path="./output/montage.png",
        normalize=True
    ):
        """
        Save montage of projection series.

        Parameters
        ----------
        series : ndarray
            Shape = (Ny, Nx, Nproj)

        out_path : str
            Output image path

        normalize : bool
            Whether to normalize each projection
        """

        os.makedirs(
            os.path.dirname(out_path),
            exist_ok=True
        )

        Ny, Nx, Nproj = series.shape

        # Projection selection
        if Nproj <= self.n_show:

            sel_idx = np.arange(Nproj)

            n_show = len(sel_idx)

        else:

            sel_idx = np.linspace(
                0,
                Nproj - 1,
                self.n_show
            ).astype(int)

            n_show = self.n_show

        n_rows = math.ceil(
            n_show / self.n_cols
        )

        fig, axes = plt.subplots(
            n_rows,
            self.n_cols,
            figsize=(
                self.n_cols * 2.2,
                n_rows * 2.2
            )
        )

        axes = np.atleast_2d(axes)

        for ax in axes.flat:
            ax.axis("off")

        if normalize:
            series = self.normalize_per_projection(
                series
            )

        for k, i in enumerate(sel_idx):

            r = k // self.n_cols
            c = k % self.n_cols

            ax = axes[r, c]

            img = series[:, :, i]

            ax.imshow(
                img,
                cmap=self.cmap
            )

            angle_deg = np.rad2deg(
                self.angles_rad[i]
            )

            ax.set_title(
                f"{angle_deg:+.1f}°",
                fontsize=8
            )

        plt.tight_layout(pad=0.5)

        plt.savefig(
            out_path,
            dpi=200
        )

        plt.close(fig)

        print(
            f"[reproj-check] montage saved to: "
            f"{out_path}"
        )