import h5py
import numpy as np
import cupy as cp
from scipy.ndimage import rotate
import pyvista as pv
import os

# -------------------------------------------------------------------------------------------------
# 1. 读取 aligned 数据
# -------------------------------------------------------------------------------------------------
def load_aligned_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        aligned_series = f['tiltSeries'][:].astype(np.float32)
        tilt_angles = f['tiltAngles'][:].astype(np.float32)
        rotation_axis_point = f['geometry/rotation_axis_point'][:] if 'geometry/rotation_axis_point' in f['geometry'] else None
        rotation_axis_vector = f['geometry/rotation_axis_vector'][:] if 'geometry/rotation_axis_vector' in f['geometry'] else None
    print(f"[load] Loaded aligned series shape={aligned_series.shape}, number of tilt angles={tilt_angles.shape[0]}")
    return aligned_series, tilt_angles, rotation_axis_point, rotation_axis_vector

# -------------------------------------------------------------------------------------------------
# 2. GPU FBP 修正版
# -------------------------------------------------------------------------------------------------
def coarse_3d_reconstruction_gpu(
    aligned_series,
    tilt_angles,
    rotation_axis_point=None,
    rotation_axis_vector=None,
    stride=1,
    Nz_target=None,
    cmap='gray',
    visualize=True,
    filter_name='ram-lak',
    interp_order=1
):
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

    print(f"[recon][GPU] Coarse 3D reconstruction: volume shape = ({Nx_ds}, {Ny_ds}, {Nz})")

    # FFT 频率 & 滤波器
    freqs = cp.fft.fftfreq(Ny_ds).astype(cp.float32)
    filter_name = filter_name.lower()
    if filter_name in ['ram-lak', 'ramp']:
        filter_vals = 2 * cp.abs(freqs)
    elif filter_name == 'shepp-logan':
        filter_vals = 2 * cp.abs(freqs) * cp.sinc(freqs / 2)
    elif filter_name == 'cosine':
        filter_vals = 2 * cp.abs(freqs) * cp.cos(cp.pi * freqs / 2)
    elif filter_name == 'hamming':
        filter_vals = 2 * cp.abs(freqs) * (0.54 + 0.46 * cp.cos(cp.pi * freqs))
    elif filter_name == 'hann':
        filter_vals = 2 * cp.abs(freqs) * (0.5 + 0.5 * cp.cos(cp.pi * freqs))
    else:
        raise ValueError(f"[recon][GPU] Unknown filter: {filter_name}")

    # 将 tilt series 放到 GPU
    aligned_gpu = cp.asarray(aligned_ds, dtype=cp.float32)
    tilt_angles_rad = cp.deg2rad(cp.asarray(tilt_angles, dtype=cp.float32))
    volume = cp.zeros((Nx_ds, Ny_ds, Nz), dtype=cp.float32)

    # 逐层 FBP
    for iz in range(Nz):
        z_idx = int(iz / Nz * Nx_ds)
        proj = aligned_gpu[z_idx, :, :]  # shape=(Ny_ds, Nproj)

        # FFT + 滤波
        proj_fft = cp.fft.fft(proj, axis=0)
        proj_fft *= filter_vals[:, None]
        proj_filt = cp.fft.ifft(proj_fft, axis=0).real.astype(cp.float32)

        # 初始化 slice
        recon_slice = cp.zeros((Nx_ds, Ny_ds), dtype=cp.float32)

        # 每条投影反投影
        for i in range(Nproj_ds):
            angle = float(tilt_angles[i])
            row = cp.asnumpy(proj_filt[:, i])
            # tile 到 XY 平面
            slice_i = np.tile(row[:, None], (1, Ny_ds))
            # 精细插值
            slice_i_rot = rotate(slice_i, angle=-angle, reshape=False, order=interp_order, mode='nearest')
            recon_slice += cp.asarray(slice_i_rot, dtype=cp.float32)

        volume[:, :, iz] = recon_slice

        if iz % 10 == 0 or iz == Nz-1:
            print(f"[recon][GPU] Reconstructed slice {iz+1}/{Nz}")

    volume_cpu = cp.asnumpy(volume)

    # 可视化
    if visualize:
        print("[recon][GPU] Visualizing volume using PyVista")
        pv.set_plot_theme("document")
        vol_pv = pv.wrap(volume_cpu)
        p = pv.Plotter()
        p.add_volume(vol_pv, cmap=cmap, opacity='linear', shade=True)
        if rotation_axis_point is not None and rotation_axis_vector is not None:
            line_points = np.array([
                rotation_axis_point - 0.5*np.array(rotation_axis_vector),
                rotation_axis_point + 0.5*np.array(rotation_axis_vector)
            ])
            p.add_lines(line_points, color='red', width=3)
        p.show()

    return volume_cpu

# -------------------------------------------------------------------------------------------------
# 3. 保存体积
# -------------------------------------------------------------------------------------------------
def save_volume_h5(volume, output_file='volume.h5'):
    print(f"[save] Saving volume to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('volume', data=volume)
    print("[save] Done")

# -------------------------------------------------------------------------------------------------
# 4. 测试函数
# -------------------------------------------------------------------------------------------------
def test_reconstruction_gpu(aligned_h5_file, stride=2, Nz_target=100, visualize=True, filter_name='ram-lak', interp_order=1):
    aligned_series, tilt_angles, rotation_axis_point, rotation_axis_vector = load_aligned_h5(aligned_h5_file)
    volume = coarse_3d_reconstruction_gpu(
        aligned_series,
        tilt_angles,
        rotation_axis_point=rotation_axis_point,
        rotation_axis_vector=rotation_axis_vector,
        stride=stride,
        Nz_target=Nz_target,
        visualize=visualize,
        filter_name=filter_name,
        interp_order=interp_order
    )
    save_volume_h5(volume, output_file='volume_gpu.h5')
    return volume

# -------------------------------------------------------------------------------------------------
# 5. 使用示例
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aligned_file = r"D:/Python/Tomography/output/aligned_2.h5"
    vol = test_reconstruction_gpu(
        aligned_file,
        stride=4,
        Nz_target=100,
        visualize=False,
        filter_name='ram-lak',
        interp_order=1
    )
