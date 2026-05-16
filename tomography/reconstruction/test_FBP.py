import h5py
import numpy as np
from skimage.transform import iradon, resize
import pyvista as pv
from scipy.ndimage import zoom
import os

# -------------------------------------------------------------------------------------------------
# 1. 读取 aligned 数据
# -------------------------------------------------------------------------------------------------
def load_aligned_h5(h5_file):
    """
    Load aligned tilt series from HDF5
    Returns:
        aligned_series: (Nx, Ny, Nproj)
        tilt_angles: (Nproj,)
        rotation_axis_point: optional, (3,)
        rotation_axis_vector: optional, (3,)
    """
    with h5py.File(h5_file, 'r') as f:
        aligned_series = f['tiltSeries'][:]
        tilt_angles = f['tiltAngles'][:]
        rotation_axis_point = f['geometry/rotation_axis_point'][:] if 'geometry/rotation_axis_point' in f['geometry'] else None
        rotation_axis_vector = f['geometry/rotation_axis_vector'][:] if 'geometry/rotation_axis_vector' in f['geometry'] else None
    print(f"[load] Loaded aligned series shape={aligned_series.shape}, number of tilt angles={tilt_angles.shape[0]}")
    return aligned_series, tilt_angles, rotation_axis_point, rotation_axis_vector

# -------------------------------------------------------------------------------------------------
# 2. 粗 3D 重建（FBP + 可视化）
# -------------------------------------------------------------------------------------------------
def coarse_3d_reconstruction(
    aligned_series,
    tilt_angles,
    rotation_axis_point=None,
    rotation_axis_vector=None,
    stride=1,
    Nz_target=None,
    cmap='gray',
    visualize=True
):
    """
    Coarse 3D reconstruction using simple FBP
    """
    Nx, Ny, Nproj = aligned_series.shape
    if Nz_target is None:
        Nz = Nx
    else:
        Nz = Nz_target

    # 下采样 tilt series
    aligned_ds = aligned_series[::stride, ::stride, :]
    Nx_ds, Ny_ds, Nproj_ds = aligned_ds.shape

    if Nproj_ds != len(tilt_angles):
        raise ValueError(f"[recon] Number of projections {Nproj_ds} does not match length of tilt_angles {len(tilt_angles)}")

    print(f"[recon] Coarse 3D reconstruction: volume shape = ({Nx_ds}, {Ny_ds}, {Nz})")

    volume = np.zeros((Nx_ds, Ny_ds, Nz), dtype=np.float32)

    # 逐层反投影
    for iz in range(Nz):
        # 简化假设：每层对应 tilt series 中同一行
        z_idx = int(iz / Nz * Nx_ds)
        projections = aligned_ds[z_idx, :, :]  # shape = (Ny_ds, Nproj_ds)
        # skimage >=0.25 要求 projections shape = (detector_pixels, n_angles)
        # tilt_angles 长度 = n_angles
        recon_slice = iradon(
            projections,
            theta=tilt_angles,
            circle=False,
            filter_name=None
        )
        # 如果 Nx_ds != Ny_ds，则 resize
        if recon_slice.shape[0] != Nx_ds or recon_slice.shape[1] != Ny_ds:
            recon_slice = resize(recon_slice, (Nx_ds, Ny_ds), order=1, mode='constant', anti_aliasing=True)
        volume[:, :, iz] = recon_slice

        if iz % 10 == 0 or iz == Nz-1:
            print(f"[recon] Reconstructed slice {iz+1}/{Nz}")

    # 可视化
    if visualize:
        print("[recon] Visualizing volume using PyVista")
        pv.set_plot_theme("document")
        vol_pv = pv.wrap(volume)
        p = pv.Plotter()
        p.add_volume(vol_pv, cmap=cmap, opacity='linear', shade=True)
        if rotation_axis_point is not None and rotation_axis_vector is not None:
            line_points = np.array([
                rotation_axis_point - 0.5*np.array(rotation_axis_vector),
                rotation_axis_point + 0.5*np.array(rotation_axis_vector)
            ])
            p.add_lines(line_points, color='red', width=3)
        p.show()

    return volume

# -------------------------------------------------------------------------------------------------
# 3. 保存体积到 HDF5
# -------------------------------------------------------------------------------------------------
def save_volume_h5(volume, output_file='volume.h5'):
    print(f"[save] Saving volume to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('volume', data=volume)
    print("[save] Done")

# -------------------------------------------------------------------------------------------------
# 4. 测试函数
# -------------------------------------------------------------------------------------------------
def test_reconstruction(aligned_h5_file, stride=2, Nz_target=100, visualize=True):
    aligned_series, tilt_angles, rotation_axis_point, rotation_axis_vector = load_aligned_h5(aligned_h5_file)
    volume = coarse_3d_reconstruction(
        aligned_series,
        tilt_angles,
        rotation_axis_point=rotation_axis_point,
        rotation_axis_vector=rotation_axis_vector,
        stride=stride,
        Nz_target=Nz_target,
        visualize=visualize
    )
    save_volume_h5(volume, output_file='volume_test.h5')
    return volume

# -------------------------------------------------------------------------------------------------
# 5. 使用示例
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aligned_file = r"D:/Python/Tomography/output/aligned_2.h5"
    vol = test_reconstruction(aligned_file, stride=2, Nz_target=500, visualize=False)
