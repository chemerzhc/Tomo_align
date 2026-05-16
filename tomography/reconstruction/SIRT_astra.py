import h5py
import numpy as np
import astra
from scipy.ndimage import zoom
import os

# ---------------------------
# 1. 读取对齐并后处理后的数据
# ---------------------------
aligned_file = r"D:/Python/Tomography/output/test_20260124_SIRT_Phasecor_aligned.h5" 
with h5py.File(aligned_file, 'r') as f:
    tilt_series = f['tiltSeries'][:].astype(np.float32) # (Nz_orig, Ndet, Nproj)
    tilt_angles = f['tiltAngles'][:].astype(np.float32)

Nz_orig, Ndet, Nproj = tilt_series.shape
print(f"[load] Loaded processed series: {tilt_series.shape}")

# ---------------------------
# 2. 统一比例下采样 (确保 X=Y=Z)
# ---------------------------
# 只需要设置一个下采样比例，例如 0.5 表示尺寸减半
downsample_ratio = 1

# 计算目标尺寸（保持各向同性）
target_side = int(Ndet * downsample_ratio)
target_height = int(Nz_orig * downsample_ratio)

# 对投影序列进行等比例缩放
tilt_series_ds = zoom(tilt_series, (downsample_ratio, downsample_ratio, 1), order=1)
Nz_ds, Ndet_ds, _ = tilt_series_ds.shape

print(f"[ds] Isotropic Ratio: {downsample_ratio}")
print(f"[ds] Downsampled Volume will be: {Ndet_ds}x{Ndet_ds}x{Nz_ds}")

# ---------------------------
# 3. 定义 ASTRA 几何
# ---------------------------
angles_rad = np.deg2rad(tilt_angles)
proj_geom = astra.create_proj_geom('parallel', 1.0, Ndet_ds, angles_rad)
vol_geom = astra.create_vol_geom(Ndet_ds, Ndet_ds)

# ---------------------------
# 4. GPU SIRT 重建 (Z-stack 模式)
# ---------------------------
# 先以 (Z, Y, X) 顺序构建，因为 ASTRA 是逐层(Z)生成的
volume_zyx = np.zeros((Nz_ds, Ndet_ds, Ndet_ds), dtype=np.float32)
num_iter = 1000 

print(f"[SIRT] Starting reconstruction of {Nz_ds} slices...")

for iz in range(Nz_ds):
    sinogram = tilt_series_ds[iz, :, :] 
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram.T)
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = {'MinConstraint': 0.0}
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    
    volume_zyx[iz, :, :] = astra.data2d.get(rec_id)
    
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    
    if iz % 50 == 0 or iz == Nz_ds - 1:
        print(f"[SIRT] Progress: {iz+1}/{Nz_ds}")

# ---------------------------
# 5. 调整维度顺序为 (X, Y, Z) 并保存
# ---------------------------
# np.transpose(2, 1, 0) 将 (Z, Y, X) 变为 (X, Y, Z)
volume_xyz = np.transpose(volume_zyx, (2, 1, 0))

output_file = r"D:/Python/Tomography/output/test_1216_recon_XYZ.h5"
with h5py.File(output_file, 'w') as f:
    f.create_dataset('volume', data=volume_xyz)

print(f"[save] Saved SIRT volume (XYZ) to {output_file}")
print(f"[info] Final Volume Shape: {volume_xyz.shape}")