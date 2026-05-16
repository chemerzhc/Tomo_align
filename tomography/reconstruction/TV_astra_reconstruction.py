import h5py
import numpy as np
import astra
from scipy.ndimage import zoom
import os
import matplotlib.pyplot as plt
import imageio

# ---------------------------
# 1. 路径与配置 (完全兼容你的文件地址)
# ---------------------------
input_h5 = r"D:/Python/Tomography/output/test_1216_processed_SIRT_REPRO.h5" 
output_h5 = r"D:/Python/Tomography/output/test_1216_recon_bwr.h5"
output_dir = r"D:/Python/Tomography/output/"

# ---------------------------
# 2. 读取与预处理 (适配对齐结果)
# ---------------------------
print(f"[recon] Loading aligned data from: {input_h5}")
with h5py.File(input_h5, 'r') as f:
    # 假设输入 aligned_series 形状为 (Nx, Ny, Nproj)
    tilt_series = f['tiltSeries'][:].astype(np.float32)
    tilt_angles = f['tiltAngles'][:] if 'tilt_angles' in f else np.linspace(-60, 60, tilt_series.shape[2])

# 下采样以适配显存 (2048 -> 512)
downsample_ratio = 0.25 
tilt_series_ds = zoom(tilt_series, (downsample_ratio, downsample_ratio, 1), order=1)
Nx_ds, Ny_ds, Nproj = tilt_series_ds.shape
print(f"[recon] Downsampled Shape: {tilt_series_ds.shape}")

# ---------------------------
# 3. 准备 ASTRA 3D 几何 (核心格式修复)
# ---------------------------
# ASTRA SIRT3D_CUDA 期望投影格式: (det_rows, num_angles, det_cols)
# 映射到我们的维度: (Ny_ds, Nproj, Nx_ds)
projections_3d = np.transpose(tilt_series_ds, (1, 2, 0)) 
projections_3d = np.ascontiguousarray(projections_3d)

angles_rad = np.deg2rad(tilt_angles)
# 几何参数: (类型, 探测器行间距, 列间距, 探测器行数, 列数, 角度)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, Ny_ds, Nx_ds, angles_rad)
# 体积参数: (X_size, Y_size, Z_size)
vol_geom = astra.create_vol_geom(Nx_ds, Nx_ds, Ny_ds)

# ---------------------------
# 4. 执行 SIRT3D 重构
# ---------------------------
print(f"[recon] Initializing ASTRA SIRT3D_CUDA...")
projections_id = astra.data3d.create('-proj3d', proj_geom, projections_3d)
reconstruction_id = astra.data3d.create('-vol', vol_geom, 0)

cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ProjectionDataId'] = projections_id
cfg['ReconstructionDataId'] = reconstruction_id
cfg['option'] = {'MinConstraint': 0.0} # 物理约束，等效于TV的部分正则化效果

alg_id = astra.algorithm.create(cfg)

# 监控参数
total_iter = 100
check_interval = 10
evolution_slices = []
residuals = []
mid_y = Ny_ds // 2 # 预览切片位置

print(f"{'Iteration':<15} | {'Residual (L2 Norm)':<20}")
print("-" * 40)

for i in range(0, total_iter, check_interval):
    astra.algorithm.run(alg_id, check_interval)
    
    # 记录残差
    res = astra.algorithm.get_res_norm(alg_id)
    residuals.append(res)
    
    # 提取预览切片 (X-Z 平面，即中心横截面)
    current_vol = astra.data3d.get(reconstruction_id)
    evolution_slices.append(current_vol[:, mid_y, :].copy())
    
    print(f"{i + check_interval:<15} | {res:<20.6e}")

# 获取最终重构体积
volume_out = astra.data3d.get(reconstruction_id)

# 显存清理
astra.algorithm.delete(alg_id)
astra.data3d.delete(projections_id)
astra.data3d.delete(reconstruction_id)

# ---------------------------
# 5. 可视化 (bwr) 与 H5 保存
# ---------------------------

# A. 保存体积
with h5py.File(output_h5, 'w') as f:
    f.create_dataset('volume', data=volume_out, compression='gzip', compression_opts=4)
    print(f"[Done] Volume saved to: {output_h5}")

# B. 残差曲线
plt.figure(figsize=(8, 4))
plt.plot(range(check_interval, total_iter + 1, check_interval), residuals, 'r-o', linewidth=2)
plt.yscale('log')
plt.title("SIRT Convergence Curve")
plt.xlabel("Iterations")
plt.ylabel("L2 Residual Norm")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "residual_curve_bwr.png"))

# C. 最终切片预览 (bwr 色图)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# X-Y 面
im0 = axes[0].imshow(volume_out[Ny_ds//2, :, :], cmap='bwr')
axes[0].set_title("X-Y Plane (Top View) - bwr")
plt.colorbar(im0, ax=axes[0])

# X-Z 面 (观察缺失角伪影)
im1 = axes[1].imshow(volume_out[:, Ny_ds//2, :], cmap='bwr')
axes[1].set_title("X-Z Plane (Side View) - bwr")
plt.colorbar(im1, ax=axes[1])

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_preview_bwr.png"))
plt.show()

# D. GIF 演化图 (bwr)
frames = []
for s in evolution_slices:
    # 归一化以适应颜色图
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-7)
    # 使用 cmap 转换为彩色
    colored_frame = plt.cm.bwr(s_norm)
    frames.append((colored_frame[:, :, :3] * 255).astype(np.uint8))

imageio.mimsave(os.path.join(output_dir, "recon_evolution_bwr.gif"), frames, fps=5)

print("[Done] Results and bwr visualizations complete.")