from tomography.alignment.Iterative_alignment import *


def run_alignment_pipeline(
    h5_file,
    output_file='aligned.h5',
    movie_file='tilt_series_rotation.mp4',
    montage_path='./output/alignment_montage.png',
    max_iter_centroid=10,
    lr_centroid=0.5,
    tol_centroid=0.5,
    max_iter_cc=20,
    lr_cc=0.5,
    cc_window_ratio=0.5,
    tol_cc=0.02,
    save_iter_frames_dir=None,
    centroid_iter_use_gpu=True,
    scan_size=0.9,
    use_fft_cc=True
):
    """
    Run tilt-series alignment pipeline:
    1. Centroid-based coarse alignment
    2. Central-window cross-correlation fine alignment
       (with 8-sample visualization)
    """

    if save_iter_frames_dir is not None:
        os.makedirs(save_iter_frames_dir, exist_ok=True)
        print(
            f"[pipeline] Visualization frames will be saved to: "
            f"{save_iter_frames_dir}"
        )

    print(f"[pipeline] Loading tilt series from: {h5_file}")

    tilt_series, tilt_angles = load_tilt_series(h5_file)

    # Uncomment if input data shape is (Nproj, Nx, Ny)
    # tilt_series = tilt_series.transpose(1, 2, 0)

    print(
        f"[pipeline] Data shape = {tilt_series.shape}, "
        f"Ntilt = {tilt_angles.size}"
    )

    result = iterative_alignment(
        tilt_series,
        tilt_angles,

        # Stage 1 parameters
        max_iter_centroid=max_iter_centroid,
        lr_centroid=lr_centroid,
        tol_centroid=tol_centroid,
        centroid_iter_use_gpu=centroid_iter_use_gpu,

        # Stage 2 parameters
        max_iter_cc=max_iter_cc,
        lr_cc=lr_cc,
        tol_cc=tol_cc,
        cc_window_ratio=cc_window_ratio,
        save_iter_frames_dir=save_iter_frames_dir,

        # Performance parameters
        use_fft_cc=True,
        scan_size=scan_size
    )

    processed_series = post_process_aligned_series(
        aligned_series=result["aligned_series"],
        tilt_angles=tilt_angles,
        rotation_axis_vector=result["rotation_axis_vector"],
        scan_size=scan_size,
        cmap="bwr",
        montage_path="./output/centroid_montage.png"
    )

    visualize_tilt_series_rotation(
        processed_series=processed_series,
        tilt_angles=tilt_angles,
        save_path=movie_file
    )

    save_aligned_h5(
        processed_series=processed_series,
        shifts_array=result["shifts"],
        tilt_angles=tilt_angles,
        output_file=output_file
    )