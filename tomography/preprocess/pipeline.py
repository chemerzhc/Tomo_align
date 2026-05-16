from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tomography.preprocess.emd_reader import load_emd_frames_and_angle
from tomography.preprocess.frame_align import align_and_sum_frames, visualize_shifts
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
    修正版 pipeline:
    - 对齐并求和每个 EMD 文件
    - 只对前三个文件可视化 shift
    - tilt angle 取第一个
    - 保存到 HDF5
    """
    emd_dir = Path(emd_dir)
    emd_files = list(emd_dir.glob("*.emd"))

    angles = []
    projections = []

    for idx, f in enumerate(tqdm(emd_files, desc="Processing EMDs")):
        frames, angle_array = load_emd_frames_and_angle(f)  # 返回 shape (n_frames,)
        angle = angle_array[0]  # 取第一个 tilt angle
        angle_mean = np.mean(angle_array)  
        print(f"\nFile {idx+1}/{len(emd_files)}: {f.name}")
        print("  Original frames shape:", frames.shape)
        print("  Tilt angle (taken first):", angle)
        print("  Tilt angle mean:", angle_mean)

        if frames.shape[0] != n_frames:
            raise ValueError(f"{f} has {frames.shape[0]} frames")

        # 对齐并求和
        summed, shifts_list = align_and_sum_frames(frames)
        print("  Summed shape after align_and_sum_frames:", summed.shape)

        # 可视化前三个文件的 shift
        if DCF_debug and idx < 3:
            visualize_shifts(shifts_list)

        # 归一化
        if normalize:
            summed = robust_normalize(summed, norm_percentile)
            print("  Summed shape after normalization:", summed.shape)

        # 保存结果
        projections.append(summed)
        angles.append(angle)
        print("  Current projections list length:", len(projections))

    # 转为 numpy array
    angles = np.asarray(angles)
    projections = np.stack(projections, axis=0)
    print("Final angles shape:", angles.shape)         # (n_files,)
    print("Final projections shape:", projections.shape)  # (n_files, H, W)

    # ---- sort by tilt angle ----
    order = np.argsort(angles)
    angles = angles[order]
    projections = projections[order]

    # 保存 HDF5
    write_tomo_hdf5(
        output_h5,
        angles,
        projections,
        attrs={
            "normalization": f"robust_p{norm_percentile}",
            "n_frames": n_frames,
            "detector": "HAADF",
            "source": "hyperspy emd",
        },
    )
