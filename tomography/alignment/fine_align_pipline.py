from tomography.alignment.Iterative_alignment import *
from tomography.alignment.SIRT_BP_alignment import *


def load_aligned_h5(h5_file):
    """
    Load coarse-aligned HDF5 file saved by save_aligned_h5()

    Returns
    -------
    aligned_series : ndarray
        shape = (Nx, Ny, Nproj)
    tilt_angles : ndarray
        shape = (Nproj,)
    shifts : ndarray or None
    rotation_axis_point : ndarray, shape = (3,)
    rotation_axis_vector : ndarray, shape = (3,)
    """

    print(f"[load] Loading aligned H5: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        aligned_series = f['tiltSeries'][:].astype(np.float32)
        tilt_angles = f['tiltAngles'][:].astype(np.float32)

        # shifts may be None or empty
        if 'shifts' in f:
            shifts = f['shifts'][:]
        else:
            shifts = None

        # geom = f['geometry']
        # rotation_axis_point = geom['rotation_axis_point'][:].astype(np.float32)
        # rotation_axis_vector = geom['rotation_axis_vector'][:].astype(np.float32)

    print(
        f"[load] tiltSeries shape = {aligned_series.shape}, "
        f"Ntilt = {tilt_angles.size}"
    )

    return (
        aligned_series,
        tilt_angles,
        shifts,
        # rotation_axis_point,
        # rotation_axis_vector
    )


def run_pc_refinement_pipeline(
    aligned_h5_file,
    output_file='aligned_refined.h5',

    # Reconstruction
    Nx_target=2048,
    Ny_target=2048,
    Nz_target=500,
    recon_algorithm="SIRT",
    recon_iter=100,
    use_gpu=True,

    # Projection-consistency refinement
    pc_outer_loops=3,
    pc_inner_iter=8,
    pc_lr=0.25,
    pc_hp_cutoff=0.15,
    pc_tol=1e-3,

    final_movie_file='final_movie_align_vis.mp4',

    verbose=True
):
    """
    Alternating SIRT ↔ Projection Consistency refinement.
    """

    if verbose:
        print(f"[PC-pipeline] Loading aligned H5: {aligned_h5_file}")

    # v2 no longer requires rotation axis and rotation center
    # since the rotation axis is now fixed along y

    # aligned_series, tilt_angles, _, axis_pt, axis_vec = load_aligned_h5(
    #     aligned_h5_file
    # )

    aligned_series, tilt_angles, _ = load_aligned_h5(
        aligned_h5_file
    )

    if verbose:
        print(f"[PC-pipeline] tilt_series = {aligned_series.shape}")

    # ============================================================
    # Outer loop
    # ============================================================
    for outer in range(pc_outer_loops):
        if verbose:
            print(
                f"\n[PC-pipeline] Outer loop "
                f"{outer + 1}/{pc_outer_loops}"
            )

        # ---------- Reconstruction ----------
        volume = astra_slice_reconstruction(
            tilt_series=aligned_series,
            tilt_angles=tilt_angles,
            Nx_target=Nx_target,
            Ny_target=Ny_target,
            Nz_target=Nz_target,
            algorithm=recon_algorithm,
            num_iter=recon_iter,
            use_gpu=use_gpu,
            verbose=verbose
        )

        # ---------- Forward projection ----------
        reproj_series = forward_project_volume(
            volume,
            np.deg2rad(tilt_angles)
        )

        if verbose:
            print(f"[PC-pipeline] reproj = {reproj_series.shape}")

        # ---------- QUICK CHECK: reprojection montage ----------
        montage_handler = ProjectionMontage(
            angles_rad=np.deg2rad(tilt_angles),
            n_show=48,
            n_cols=8,
            cmap="bwr"
        )

        # ---------- Dimension check and resampling ----------
        # Ensure aligned_series and reproj_series
        # do not exceed the target dimensions
        #
        # target_v corresponds to Nz_target
        # target_u corresponds to Nx_target

        if verbose:
            print(
                "[astraSIRT] Running dimension check before "
                "projection-consistency refinement..."
            )

        aligned_series = montage_handler.ensure_dimension_match(
            aligned_series,
            Nz_target,
            Nx_target,
            verbose=verbose
        )

        reproj_series = montage_handler.ensure_dimension_match(
            reproj_series,
            Nz_target,
            Nx_target,
            verbose=verbose
        )

        # Force exact shape match to avoid
        # one-pixel mismatch from rounding

        if aligned_series.shape != reproj_series.shape:
            if verbose:
                print(
                    f"[astraSIRT] Force-matching dimensions: "
                    f"{reproj_series.shape} -> "
                    f"{aligned_series.shape}"
                )

            reproj_series = zoom(
                reproj_series,
                (
                    aligned_series.shape[0] / reproj_series.shape[0],
                    aligned_series.shape[1] / reproj_series.shape[1],
                    1
                ),
                order=1
            )

        # ---------- NORMALIZATION ----------
        aligned_series = montage_handler.normalize_per_projection(
            aligned_series
        )

        reproj_series = montage_handler.normalize_per_projection(
            reproj_series
        )

        # ---------- QUICK CHECK: SAVE MONTAGES ----------
        if verbose:

            # Save normalized reprojection series
            montage_handler.save_montage(
                reproj_series,
                out_path="./output/reproj_montage.png",
                normalize=False
            )

            # Save normalized aligned tilt series
            montage_handler.save_montage(
                aligned_series,
                out_path="./output/aligned_series_montage.png",
                normalize=False
            )

        # ---------- Projection-consistency refinement ----------
        aligned_series = projection_consistency_refinement(
            tilt_series=aligned_series,
            reproj_series=reproj_series,
            max_iter=pc_inner_iter,
            lr=pc_lr,
            hp_cutoff=pc_hp_cutoff,
            tol=pc_tol
        )

    # ============================================================
    # Save
    # ============================================================
    if verbose:
        print(
            f"[PC-pipeline] Saving refined alignment "
            f"→ {output_file}"
        )

    visualize_tilt_series_rotation(
        processed_series=aligned_series,
        tilt_angles=tilt_angles,
        save_path=final_movie_file
    )

    save_aligned_h5(
        processed_series=aligned_series,
        shifts_array=None,
        tilt_angles=tilt_angles,
        output_file=output_file
    )

    return {
        "aligned_series": aligned_series,
        "tilt_angles": tilt_angles
    }