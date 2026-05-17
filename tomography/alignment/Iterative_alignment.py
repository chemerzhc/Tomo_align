import h5py
import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
from scipy.fft import fft2, ifft2
import os
import matplotlib.pyplot as plt
import imageio

# -------------------------------------------------------------------------------------------------
# Load tilt series
# -------------------------------------------------------------------------------------------------
def load_tilt_series(h5_file):

    """Load tilt series and tilt angles from an HDF5 file.
    
    Args:
        h5_file (str): Path to input HDF5 file.
    
    Returns:
        tuple: (tilt_series, tilt_angles).
    """
    print(f"[load] Loading tilt series from {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        tilt_series = f['tiltSeries'][:]
        tilt_angles = f['tiltAngles'][:]

    print(
        f"[load] Loaded tilt series shape: {tilt_series.shape}, "
        f"number of angles: {tilt_angles.shape[0]}"
    )

    return tilt_series, tilt_angles


# -------------------------------------------------------------------------------------------------
# Apply shifts to projections
# -------------------------------------------------------------------------------------------------
def apply_shifts(tilt_series, shifts_array):

    """Apply XY shifts to all projections.
    
    Args:
        tilt_series (np.ndarray): Input tilt series.
        shifts_array (np.ndarray): Shift values for each projection.
    
    Returns:
        np.ndarray: Shifted tilt series.
    """
    num_proj = tilt_series.shape[2]
    aligned_series = np.zeros_like(tilt_series)

    for i in range(num_proj):
        aligned_series[:, :, i] = nd_shift(
            tilt_series[:, :, i],
            shift=shifts_array[i]
        )

    print(f"[shift] Applied shifts to {num_proj} projections")

    return aligned_series


# -------------------------------------------------------------------------------------------------
# Compute centroids
# -------------------------------------------------------------------------------------------------
from scipy.ndimage import center_of_mass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os


def compute_centroids(tilt_series, n_threads=8):
    """
    Multi-threaded centroid computation.
    Interface unchanged.
    """

    num_proj = tilt_series.shape[2]
    centroids = np.zeros((num_proj, 2))

    if n_threads is None:
        n_threads = min(os.cpu_count(), num_proj)

    def _one_centroid(i):
        return i, center_of_mass(tilt_series[:, :, i])

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i, c in executor.map(_one_centroid, range(num_proj)):
            centroids[i] = c

    print(
        f"[centroid][MT] Computed centroids for "
        f"{num_proj} projections using {n_threads} threads"
    )

    return centroids


# -------------------------------------------------------------------------------------------------
# Two-stage iterative alignment:
# centroid coarse alignment + central-window cross-correlation fine alignment
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import center_of_mass
from scipy.signal import correlate2d
from scipy.fft import fft2, ifft2, fftshift


def iterative_alignment(
    tilt_series,
    tilt_angles,
    max_iter_centroid=10,
    max_iter_cc=20,
    lr_centroid=0.5,
    lr_cc=0.5,
    tol_centroid=0.5,
    tol_cc=0.02,
    save_iter_frames_dir=None,
    cc_window_ratio=0.5,
    centroid_iter_use_gpu=False,
    use_fft_cc=True,
    scan_size=0.8
):
    """
    Iterative alignment with centroid-based coarse alignment
    and central-window cross-correlation fine alignment.
    """

    print(
        "[align] Starting iterative alignment "
        "(centroid + central-window CC)"
    )

    num_proj = tilt_series.shape[2]

    shifts_array = np.zeros((num_proj, 2))
    aligned_series = tilt_series.copy()

    centroid_mean_shifts = []
    centroid_max_shifts = []

    # --------------------
    # Stage 1:
    # Centroid-based coarse alignment
    # --------------------
    print("[align] Stage 1: Centroid-based coarse alignment")

    if centroid_iter_use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import center_of_mass as cp_center_of_mass

            use_gpu_stage1 = True

            print(
                "[align][centroid] Using GPU (CuPy) "
                "for centroid computation"
            )

        except ImportError:
            print("[align][centroid] CuPy not found, fallback to CPU")
            use_gpu_stage1 = False

    else:
        use_gpu_stage1 = False

    for iteration in range(max_iter_centroid):

        centroids = np.zeros((num_proj, 2))

        for i in range(num_proj):

            if use_gpu_stage1:
                img_cp = cp.asarray(aligned_series[:, :, i])

                centroids[i] = cp.asnumpy(
                    cp_center_of_mass(img_cp)
                )

            else:
                centroids[i] = center_of_mass(
                    aligned_series[:, :, i]
                )

        idx = np.arange(num_proj)

        coeffs_x = np.polyfit(idx, centroids[:, 0], 1)
        coeffs_y = np.polyfit(idx, centroids[:, 1], 1)

        pred_centroids = np.column_stack((
            np.polyval(coeffs_x, idx),
            np.polyval(coeffs_y, idx)
        ))

        delta = pred_centroids - centroids

        shifts_array += lr_centroid * delta

        aligned_series = apply_shifts(
            tilt_series,
            shifts_array
        )

        max_shift = np.max(
            np.linalg.norm(delta, axis=1)
        )

        mean_shift = np.mean(
            np.linalg.norm(delta, axis=1)
        )

        centroid_mean_shifts.append(mean_shift)
        centroid_max_shifts.append(max_shift)

        print(
            f"[align][centroid] Iter "
            f"{iteration + 1}/{max_iter_centroid} | "
            f"mean Δ={mean_shift:.3f}px | "
            f"max Δ={max_shift:.3f}px"
        )

        if save_iter_frames_dir is not None:

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5)
            )

            mid_idx = num_proj // 2

            axes[0].imshow(
                aligned_series[:, :, mid_idx],
                cmap='plasma'
            )

            axes[0].plot(
                shifts_array[:, 1] + aligned_series.shape[1] // 2,
                shifts_array[:, 0] + aligned_series.shape[0] // 2,
                'ro',
                label='centroid trace'
            )

            axes[0].set_title(
                f"Centroid Iter {iteration + 1}"
            )

            axes[0].legend()

            axes[1].plot(
                range(1, iteration + 2),
                centroid_mean_shifts,
                'g-o',
                label='mean Δ'
            )

            axes[1].plot(
                range(1, iteration + 2),
                centroid_max_shifts,
                'm-o',
                label='max Δ'
            )

            axes[1].set_title(
                "Shift evolution (centroid)"
            )

            axes[1].legend()

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    save_iter_frames_dir,
                    f"centroid_iter_{iteration + 1:02d}.png"
                ),
                dpi=150
            )

            plt.close(fig)

        if max_shift < tol_centroid:
            print("[align][centroid] Converged")
            break

    # --------------------
    # Stage 2:
    # Central-window cross-correlation fine alignment
    # --------------------
    print(
        f"[align] Stage 2: Fine alignment "
        f"(FFT Optimized: {use_fft_cc})"
    )

    Nx, Ny, _ = aligned_series.shape

    win_x = int(Nx * cc_window_ratio)
    win_y = int(Ny * cc_window_ratio)

    x_start = (Nx - win_x) // 2
    y_start = (Ny - win_y) // 2

    max_shifts = []
    mean_shifts = []

    # Uniformly distributed sample projections for visualization
    sample_indices = np.linspace(
        0,
        num_proj - 1,
        8,
        dtype=int
    )

    for iteration in range(max_iter_cc):

        delta_list = []
        samples_to_plot = []

        for i in range(num_proj):

            # Reference image construction
            if i == 0:
                ref_image = aligned_series[:, :, 1]

            elif i == num_proj - 1:
                ref_image = aligned_series[:, :, i - 1]

            else:
                ref_image = 0.5 * (
                    aligned_series[:, :, i - 1]
                    + aligned_series[:, :, i + 1]
                )

            # Extract central window
            ref_patch = ref_image[
                x_start:x_start + win_x,
                y_start:y_start + win_y
            ]

            img_patch = aligned_series[
                x_start:x_start + win_x,
                y_start:y_start + win_y,
                i
            ]

            # Compute cross-correlation
            if use_fft_cc:

                # FFT-based accelerated cross-correlation
                f_ref = fft2(ref_patch)
                f_mov = fft2(img_patch)

                cc = np.real(
                    fftshift(
                        ifft2(f_ref * np.conj(f_mov))
                    )
                )

            else:

                # Spatial-domain correlation
                cc = correlate2d(
                    ref_patch,
                    img_patch,
                    mode='same'
                )

            max_idx = np.unravel_index(
                np.argmax(cc),
                cc.shape
            )

            shift_est = (
                np.array(max_idx)
                - np.array([win_x // 2, win_y // 2])
            )

            delta_list.append(shift_est)

            # Collect visualization samples
            if i in sample_indices:
                samples_to_plot.append(
                    (ref_patch, img_patch, i)
                )

        delta_array = np.array(delta_list)

        shifts_array += lr_cc * delta_array

        aligned_series = apply_shifts(
            tilt_series,
            shifts_array
        )

        max_shift = np.max(
            np.linalg.norm(delta_array, axis=1)
        )

        mean_shift = np.mean(
            np.linalg.norm(delta_array, axis=1)
        )

        max_shifts.append(max_shift)
        mean_shifts.append(mean_shift)

        print(
            f"[align][CC] Iter "
            f"{iteration + 1}/{max_iter_cc} | "
            f"mean Δ={mean_shift:.3f}px | "
            f"max Δ={max_shift:.3f}px"
        )

        # --- 8-sample patch comparison visualization ---
        if save_iter_frames_dir is not None:

            fig_samples, axes_samples = plt.subplots(
                2,
                8,
                figsize=(20, 6)
            )

            for col, (r_p, m_p, idx) in enumerate(samples_to_plot):

                axes_samples[0, col].imshow(
                    r_p,
                    cmap='gray'
                )

                axes_samples[0, col].set_title(
                    f"Ref #{idx}"
                )

                axes_samples[0, col].axis('off')

                axes_samples[1, col].imshow(
                    m_p,
                    cmap='magma'
                )

                axes_samples[1, col].set_title(
                    f"Mov #{idx}"
                )

                axes_samples[1, col].axis('off')

            plt.suptitle(
                f"CC Alignment Samples - Iteration "
                f"{iteration + 1}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    save_iter_frames_dir,
                    f"cc_samples_iter_{iteration + 1:02d}.png"
                )
            )

            plt.close(fig_samples)

            # Shift trajectory visualization
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5)
            )

            mid_idx = num_proj // 2

            axes[0].imshow(
                aligned_series[:, :, mid_idx],
                cmap='gray'
            )

            axes[0].plot(
                shifts_array[:, 1]
                + aligned_series.shape[1] // 2,

                shifts_array[:, 0]
                + aligned_series.shape[0] // 2,

                'ro',
                label='CC trace'
            )

            axes[0].set_title(
                f"CC Iter {iteration + 1}"
            )

            axes[0].legend()

            axes[1].plot(
                range(1, iteration + 2),
                mean_shifts,
                'g-o',
                label='mean Δ'
            )

            axes[1].plot(
                range(1, iteration + 2),
                max_shifts,
                'm-o',
                label='max Δ'
            )

            axes[1].set_xlabel("Iteration")

            axes[1].set_title(
                "Shift evolution (CC)"
            )

            axes[1].legend()

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    save_iter_frames_dir,
                    f"cc_iter_{iteration + 1:02d}.png"
                ),
                dpi=150
            )

            plt.close(fig)

        if max_shift < tol_cc:
            print("[align][CC] Converged")
            break

    # --------------------
    # Rotation axis 3D line fitting
    # --------------------
    z_coords = np.arange(num_proj)

    x_coords = (
        shifts_array[:, 0]
        + aligned_series.shape[0] // 2
    )

    y_coords = (
        shifts_array[:, 1]
        + aligned_series.shape[1] // 2
    )

    points = np.stack(
        [x_coords, y_coords, z_coords],
        axis=1
    )

    centroid = points.mean(axis=0)

    _, _, vh = np.linalg.svd(points - centroid)

    direction = vh[0]

    rotation_axis_point = centroid

    rotation_axis_vector = (
        direction / np.linalg.norm(direction)
    )

    print(
        f"[align] Estimated rotation axis: "
        f"{rotation_axis_point}, "
        f"{rotation_axis_vector}"
    )

    return {
        "aligned_series": aligned_series,
        "shifts": shifts_array,
        "rotation_axis_point": rotation_axis_point,
        "rotation_axis_vector": rotation_axis_vector
    }


# -------------------------------------------------------------------------------------------------
# Visualize tilt series rotation
# -------------------------------------------------------------------------------------------------
import numpy as np
from scipy.ndimage import rotate, zoom
import imageio
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def visualize_tilt_series_rotation(
    processed_series,
    tilt_angles,
    save_path='tilt_series_rotation.mp4',
    fps=15
):
    """
    Matplotlib overlay version (sharp text, publication quality)

    - avoids PIL raster text blur
    - uses vector rendering
    - supports grayscale colormap + rotation axis + tilt label
    """

    N, _, Nproj = processed_series.shape
    frames = []

    print(
        f"[viz] Creating movie. "
        f"Shape: {processed_series.shape}"
    )

    cmap = plt.get_cmap('gray')

    for i in range(Nproj):

        img = processed_series[:, :, i]

        # ---------------------------
        # Normalize
        # ---------------------------
        img_min, img_max = img.min(), img.max()

        img_norm = (
            (img - img_min)
            / (img_max - img_min + 1e-6)
        )

        img_rgba = cmap(img_norm)
        img_rgb = img_rgba[:, :, :3]

        # ---------------------------
        # Matplotlib figure canvas
        # ---------------------------
        fig = plt.figure(
            figsize=(5, 5),
            dpi=200
        )

        canvas = FigureCanvas(fig)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(
            img_rgb,
            interpolation='nearest'
        )

        ax.axis('off')

        # ---------------------------
        # Rotation axis
        # ---------------------------
        cx = N // 2

        ax.axvline(
            cx,
            color='white',
            linewidth=1.5,
            linestyle='--',
            alpha=0.9
        )

        # ---------------------------
        # Tilt angle label
        # ---------------------------
        ax.text(
            0.03,
            0.95,
            f"Tilt Angle: {tilt_angles[i]:.1f}°",
            transform=ax.transAxes,
            color='white',
            fontsize=0.1 * N,
            va='top',
            ha='left',
            bbox=dict(
                facecolor='black',
                alpha=0.3,
                edgecolor='none'
            )
        )

        # ---------------------------
        # Render to numpy
        # ---------------------------
        canvas.draw()

        buf = np.asarray(canvas.buffer_rgba())

        frame = buf[:, :, :3].copy()

        frame = frame.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        frames.append(frame)

        plt.close(fig)

        if i % 50 == 0 or i == Nproj - 1:
            print(
                f"[viz] Rendering frame "
                f"{i + 1}/{Nproj}"
            )

    # ---------------------------
    # Save video
    # ---------------------------
    imageio.mimsave(
        save_path,
        frames,
        fps=fps,
        codec='libx264'
    )

    print(f"[viz] Movie saved to: {save_path}")


# -------------------------------------------------------------------------------------------------
# Alignment postprocess function
# Align the rotation axis with the y-axis
# -------------------------------------------------------------------------------------------------
from scipy.ndimage import rotate
import numpy as np
import matplotlib.pyplot as plt
import os


def post_process_aligned_series(
    aligned_series,
    tilt_angles,
    rotation_axis_vector,
    scan_size=0.8,
    cmap="gray",
    montage_path="./output/montage_120.png"
):
    """
    Post-processing pipeline:
    1. Rotation alignment
    2. Square cropping
    3. 120-frame montage generation with tilt labels
    """

    Nx, Ny, Nproj = aligned_series.shape

    vx, vy, _ = rotation_axis_vector

    # ==============================
    # Compute rotation angle
    # ==============================
    v = np.array([vx, vy], dtype=float)

    if np.linalg.norm(v) == 0:
        angle_to_rotate = 0.0

    else:
        v /= np.linalg.norm(v)

        target = np.array([0.0, 1.0])

        angle_to_rotate = np.degrees(
            np.arctan2(
                v[0] * target[1] - v[1] * target[0],
                np.dot(v, target)
            )
        )

    print(
        f"[post-process] Starting post-processing "
        f"for {Nproj} projections"
    )

    print(
        f"[post-process] Rotation axis vector: "
        f"[{vx:.3f}, {vy:.3f}], "
        f"Rotate angle: {angle_to_rotate:.2f}°"
    )

    # ==============================
    # Square crop parameters
    # ==============================
    side = int(min(Nx, Ny) * scan_size)

    cx, cy = Nx // 2, Ny // 2

    r = side // 2

    print(
        f"[post-process] Cropping to square: "
        f"{side}x{side} "
        f"(scan_size={scan_size})"
    )

    processed_series = np.zeros(
        (side, side, Nproj),
        dtype=aligned_series.dtype
    )

    # ==============================
    # Rotation + cropping
    # ==============================
    for i in range(Nproj):

        img_rot = rotate(
            aligned_series[:, :, i],
            angle_to_rotate,
            reshape=False,
            order=1
        )

        processed_series[:, :, i] = img_rot[
            cx - r:cx - r + side,
            cy - r:cy - r + side
        ]

        if i % 50 == 0 or i == Nproj - 1:
            print(
                f"[post-process] Processing frames: "
                f"{i + 1}/{Nproj}"
            )

    # ==============================
    # Generate 10x12 montage
    # ==============================
    n_rows, n_cols = 10, 12
    n_total = 120

    indices = np.linspace(
        0,
        Nproj - 1,
        n_total
    ).astype(int)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(24, 20),
        dpi=100,
        facecolor='black'
    )

    axes = axes.flatten()

    print(
        "[post-process] Generating 10x12 montage "
        "with tilt angle labels"
    )

    for i, idx in enumerate(indices):

        ax = axes[i]

        curr_img = processed_series[:, :, idx]

        v_min, v_max = (
            curr_img.min(),
            curr_img.max()
        )

        ax.imshow(
            curr_img,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max
        )

        ax.text(
            0.05,
            0.95,
            f"{tilt_angles[idx]:.1f}°",
            color='white',
            fontsize=9,
            fontweight='bold',
            transform=ax.transAxes,
            va='top',
            ha='left',
            bbox=dict(
                facecolor='black',
                alpha=0.5,
                lw=0
            )
        )

        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(
        wspace=0.01,
        hspace=0.01,
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01
    )

    os.makedirs(
        os.path.dirname(montage_path),
        exist_ok=True
    )

    plt.savefig(
        montage_path,
        bbox_inches='tight',
        facecolor='black'
    )

    plt.close()

    print(
        f"[post-process] Montage saved to: "
        f"{montage_path}"
    )

    return processed_series


# -------------------------------------------------------------------------------------------------
# Save coarse aligned data
# -------------------------------------------------------------------------------------------------
def save_aligned_h5(
    processed_series,
    shifts_array,
    tilt_angles,
    output_file
):
    """Save aligned tilt series into HDF5 file.
    
    Args:
        processed_series (np.ndarray): Processed tilt series.
        shifts_array (np.ndarray): Alignment shifts.
        tilt_angles (np.ndarray): Tilt angles.
        output_file (str): Output file path.
    """
    print(
        f"[save] Saving post-processed data "
        f"to {output_file}"
    )

    with h5py.File(output_file, 'w') as f:

        f.create_dataset(
            'tiltSeries',
            data=processed_series
        )

        f.create_dataset(
            'tiltAngles',
            data=tilt_angles
        )

        if shifts_array is not None:
            f.create_dataset(
                'shifts',
                data=shifts_array
            )

    print(
        f"[save] Done. Final shape: "
        f"{processed_series.shape}"
    )


# =============================
# Adjacent-frame cumulative XY alignment
# =============================
def common_line_alignment(
    tilt_series,
    max_iter=20,
    lr=0.7,
    tol=0.1,
    max_shift_per_iter=100.0,
    output_dir="Common_line_align_log_gpu"
):
    """Perform adjacent-frame cumulative XY alignment using VMF and GPU.
    
    Args:
        tilt_series (np.ndarray): Input tilt series.
    
    Returns:
        dict: Aligned series, shifts, and loss history.
    """
    import os
    import cupy as cp
    from scipy.ndimage import shift
    from skimage.registration import phase_cross_correlation
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    log_file = open(
        os.path.join(output_dir, "log.txt"),
        "w",
        encoding='utf-8'
    )

    tilt_series_gpu = cp.asarray(tilt_series)

    Nx, Ny, Nproj = tilt_series.shape

    shifts_array = np.zeros((Nproj, 2))

    aligned_series_gpu = tilt_series_gpu.copy()

    loss_history = []

    print("[VMF XY simultaneous GPU] Alignment started")

    log_file.write(
        "[VMF XY simultaneous GPU] Alignment started\n"
    )

    for iteration in range(max_iter):

        delta_shifts = np.zeros_like(shifts_array)

        vmf_profiles_gpu = []

        # ==============================
        # Step 1: GPU VMF
        # ==============================
        for i in range(Nproj):

            vertical_profile = cp.sum(
                aligned_series_gpu[:, :, i],
                axis=1
            )

            f = cp.fft.fft(vertical_profile)

            f[0] = 0

            vmf_profiles_gpu.append(
                cp.real(cp.fft.ifft(f))
            )

        vmf_profiles_gpu = cp.array(vmf_profiles_gpu)

        vmf_profiles = cp.asnumpy(vmf_profiles_gpu)

        # ==============================
        # Step 2: XY shift estimation
        # ==============================
        aligned_series_cpu = cp.asnumpy(
            aligned_series_gpu
        )

        for i in range(1, Nproj):

            shift_est, _, _ = phase_cross_correlation(
                aligned_series_cpu[:, :, i - 1],
                aligned_series_cpu[:, :, i],
                upsample_factor=20
            )

            dy, dx = shift_est

            dx = np.clip(
                dx,
                -max_shift_per_iter,
                max_shift_per_iter
            )

            dy = np.clip(
                dy,
                -max_shift_per_iter,
                max_shift_per_iter
            )

            delta_shifts[i, 0] = dx
            delta_shifts[i, 1] = dy

        shifts_array += lr * delta_shifts

        # ==============================
        # Step 3: Apply XY shifts
        # ==============================
        aligned_series_gpu = cp.zeros_like(
            tilt_series_gpu
        )

        for i in range(Nproj):

            shifted_cpu = shift(
                tilt_series[:, :, i],
                shift=(
                    shifts_array[i, 1],
                    shifts_array[i, 0]
                ),
                order=1,
                mode='nearest'
            )

            aligned_series_gpu[:, :, i] = cp.asarray(
                shifted_cpu
            )

        aligned_series = cp.asnumpy(
            aligned_series_gpu
        )

        # ==============================
        # Step 4: Loss computation
        # ==============================
        loss = np.mean([
            np.var(
                vmf_profiles[i]
                - vmf_profiles[i - 1]
            )
            for i in range(1, Nproj)
        ])

        loss_history.append(loss)

        max_shift = np.max(np.abs(delta_shifts))

        log_line = (
            f"Iter {iteration + 1} | "
            f"Loss={loss:.6f} | "
            f"MaxΔ={max_shift:.3f}"
        )

        print(log_line)

        log_file.write(log_line + "\n")

        # ==============================
        # Visualization
        # ==============================
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10)
        )

        mid = Nproj // 2

        axes[0, 0].imshow(
            aligned_series[:, :, mid],
            cmap='bwr'
        )

        axes[0, 0].set_title(
            f"Iteration {iteration + 1} | Mid frame"
        )

        rotation_axis_x = Nx // 2

        axes[0, 0].axvline(
            rotation_axis_x,
            color='r',
            linestyle='--',
            linewidth=2,
            label='Rotation axis'
        )

        axes[0, 0].legend()

        axes[0, 1].plot(
            vmf_profiles[mid],
            label="Mid VMF"
        )

        axes[0, 1].plot(
            vmf_profiles[mid - 1],
            label="Prev VMF"
        )

        axes[0, 1].plot(
            vmf_profiles[mid + 1],
            label="Next VMF"
        )

        axes[0, 1].legend()

        axes[0, 1].set_title(
            "VMF profiles (mid +/- 1)"
        )

        for i in range(Nproj):
            axes[1, 0].plot(
                vmf_profiles[i] + i * 0.1
            )

        axes[1, 0].set_title(
            "All VMF profiles (offset)"
        )

        axes[1, 0].set_xlabel("Y pixels")

        axes[1, 1].plot(
            shifts_array[:, 0],
            label="X shift"
        )

        axes[1, 1].plot(
            shifts_array[:, 1],
            label="Y shift"
        )

        axes[1, 1].set_title(
            "Accumulated shifts"
        )

        axes[1, 1].set_xlabel("Frame index")

        axes[1, 1].legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"iter_{iteration:02d}.png"
            )
        )

        plt.close()

        # Loss curve
        plt.figure()

        plt.plot(
            loss_history,
            marker='o'
        )

        plt.xlabel("Iteration")

        plt.ylabel("Loss")

        plt.title("Loss curve")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                output_dir,
                "loss_curve.png"
            )
        )

        plt.close()

        if max_shift < tol:
            print(
                "[VMF XY simultaneous GPU] Converged"
            )

            log_file.write(
                "[VMF XY simultaneous GPU] Converged\n"
            )

            break

    log_file.close()

    return {
        "aligned_series": aligned_series,
        "shifts": shifts_array,
        "loss_history": loss_history
    }