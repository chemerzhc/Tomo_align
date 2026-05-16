import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import matplotlib.pyplot as plt

def align_and_sum_frames(frames, upsample_factor=10):
    ref = frames[0].astype(np.float32)
    aligned = [ref]
    shifts_list = []

    for i in range(1, frames.shape[0]):
        img = frames[i].astype(np.float32)

        shift_xy, _, _ = phase_cross_correlation(
            ref,
            img,
            upsample_factor=upsample_factor,
            normalization="phase",
        )

        aligned_img = shift(
            img, shift=shift_xy, order=1, mode="reflect"
        )
        aligned.append(aligned_img)
        shifts_list.append(shift_xy)

    aligned = np.array(aligned, dtype=np.float32)
    print("aligned shape after np.array:", aligned.shape) 
    summed = np.sum(aligned, axis=0)
    print("summed shape after np.sum:", summed.shape)
    return summed, shifts_list



def visualize_shifts(shifts_list, xlim=(-5,5), ylim=(-5,5), title="Aligned Shifts"):
    """
    可视化对齐后的位移向量。
    shifts_list: list 或 array, 每个元素是 (y_shift, x_shift)
    xlim, ylim: 可视化坐标轴范围
    """
    shifts_array = np.array(shifts_list)
    plt.figure(figsize=(5,5))
    plt.quiver(
        np.zeros(len(shifts_array)), np.zeros(len(shifts_array)),
        shifts_array[:, 1], shifts_array[:, 0],  # x对应列, y对应行
        angles='xy', scale_units='xy', scale=1, color='r'
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x shift")
    plt.ylabel("y shift")
    plt.title(title)
    plt.grid(True)
    plt.show()

