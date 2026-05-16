from tomography.alignment.Iterative_alignment import *
from tomography.alignment.SIRT_BP_alignment import *
def com_align_pip(
        h5_file,
        max_iter=20,
        lr=0.8,
        output_file='Comline_aligned_test.h5',
        movie_file='after_com_line.mp4'

):
    # common-line
    tilt_series, tilt_angles=load_tilt_series(h5_file)
    result = common_line_alignment(
        tilt_series=tilt_series,
        max_iter=max_iter,
        lr=lr
    )

    aligned = result["aligned_series"]
    estimated_shifts = result["shifts"]
    visualize_tilt_series_rotation(
        processed_series=aligned,
        tilt_angles=tilt_angles,
        save_path=movie_file
    )
    save_aligned_h5(
        processed_series=result["aligned_series"],
        shifts_array=None, 
        tilt_angles=tilt_angles,
        output_file=output_file
    )
