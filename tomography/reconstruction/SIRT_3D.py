import h5py
import numpy as np
import astra
from scipy.ndimage import zoom
import os
import matplotlib.pyplot as plt
import imageio

# ---------------------------
# 1. 读取数据 (保持不变)
# ---------------------------
aligned_file = r"D:\Python\Tomography\output\Xray_misalign_test/misaligned_xray_dataset.h5" 
with h5py.File(aligned_file, 'r') as f:
    tilt_series = f['tiltSeries'][:].astype(np.float32)
    tilt_angles = f['tiltAngles'][:].astype(np.float32)

Nz_orig, Ndet_orig, Nproj = tilt_series.shape
downsample_ratio = 1.0
tilt_series_ds = zoom(tilt_series, (downsample_ratio, downsample_ratio, 1), order=1)
Nz_ds, Ndet_ds, _ = tilt_series_ds.shape

# ---------------------------
# 3. 准备 ASTRA 3D 数据与几何
# ---------------------------
projections_3d = np.transpose(tilt_series_ds, (0, 2, 1))
projections_3d = np.ascontiguousarray(projections_3d)

angles_rad = np.deg2rad(tilt_angles)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, Nz_ds, Ndet_ds, angles_rad)
vol_geom = astra.create_vol_geom(Ndet_ds, Ndet_ds, Nz_ds)

# ---------------------------
# 4. 执行带残差监控的 3D SIRT
# ---------------------------
print(f"[SIRT3D] Initializing SIRT3D_CUDA...")
projections_id = astra.data3d.create('-proj3d', proj_geom, projections_3d)
reconstruction_id = astra.data3d.create('-vol', vol_geom, 0)

cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ProjectionDataId'] = projections_id
cfg['ReconstructionDataId'] = reconstruction_id
cfg['option'] = {'MinConstraint': 0.0}

alg_id = astra.algorithm.create(cfg)

# --- 监控配置 ---
total_iter = 2000        
check_interval = 10     
evolution_slices = []   
residuals = []          # 存储残差值
mid_z = Nz_ds // 2      

print(f"{'Iteration':<15} | {'Residual (L2 Norm)':<20}")
print("-" * 40)

for i in range(0, total_iter, check_interval):
    # 执行迭代
    astra.algorithm.run(alg_id, check_interval)
    
    # 1. 记录残差
    res = astra.algorithm.get_res_norm(alg_id)
    residuals.append(res)
    
    # 2. 提取预览切片
    current_vol = astra.data3d.get(reconstruction_id)
    evolution_slices.append(current_vol[:, :, mid_z].copy())
    
    print(f"{i + check_interval:<15} | {res:<20.6e}")

# 获取最终结果
volume_out = astra.data3d.get(reconstruction_id)

# 清理内存
astra.algorithm.delete(alg_id)
astra.data3d.delete(projections_id)
astra.data3d.delete(reconstruction_id)

# ---------------------------
# 5. 可视化排版与保存
# ---------------------------

# A. 保存 H5
volume_xyz = np.transpose(volume_out, (1, 0, 2))
output_file = r"D:/Python/Tomography/output/test2D.h5"
with h5py.File(output_file, 'w') as f:
    f.create_dataset('volume', data=volume_xyz)

# B. 绘制残差曲线
plt.figure(figsize=(8, 5))
plt.plot(range(check_interval, total_iter + 1, check_interval), residuals, 'o-', color='tab:blue', linewidth=2)
plt.yscale('log') # 通常残差用对数坐标看收敛更清晰
plt.title("SIRT Convergence Curve (L2 Residual)")
plt.xlabel("Iterations")
plt.ylabel("Residual Norm (log scale)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig(r"D:/Python/Tomography/output/residual_curve0301.png")
plt.show()

# C. 矩阵排版预览 (2行 5列)
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle(f"Reconstruction Evolution (Slice Z={mid_z})", fontsize=16)
axes = axes.flatten()

for idx, slice_img in enumerate(evolution_slices):
    if idx < len(axes):
        axes[idx].imshow(slice_img, cmap='gray')
        axes[idx].set_title(f"Iter {(idx+1)*check_interval}")
        axes[idx].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(r"D:/Python/Tomography/output/evolution_matrix0301.png")
plt.show()

# D. 保存 GIF
frames = [((s - s.min()) / (s.max() - s.min() + 1e-7) * 255).astype(np.uint8) for s in evolution_slices]
imageio.mimsave(r"D:/Python/Tomography/output/recon_progress0301.gif", frames, fps=3)

print("[Done] Results, Residual Curve, and Matrix View saved.")