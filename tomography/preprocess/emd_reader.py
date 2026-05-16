from pathlib import Path
import hyperspy.api as hs
import numpy as np

def load_emd_frames_and_angle(emd_file: Path):
    """
    Load frames and tilt angle from an EMD file.

    Parameters
    ----------
    emd_file : Path
        Path to the EMD file.

    Returns
    -------
    frames : np.ndarray (n_frames, Y, X)
        Image stack.
    tilt_angles : np.ndarray (n_frames,)
        Tilt angles in degrees (replicated if only one in metadata).
    """
    if not emd_file.exists():
        raise FileNotFoundError(f"File does not exist: {emd_file}")

    # ---- load EMD safely ----
    try:
        s = hs.load(str(emd_file), load_original_metadata=True, lazy=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load EMD file {emd_file}: {e}")

    # ---- frames ----
    data = np.asarray(s.data)
    if data.ndim == 2:
        # 单帧也返回 shape (1, Y, X)
        frames = data[np.newaxis, :, :].astype(np.float32)
    elif data.ndim == 3:
        frames = data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected data shape {data.shape} in {emd_file}")

    # ---- tilt angles ----
    try:
        tilt_angle = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
    except AttributeError:
        tilt_angle = np.nan  # 没找到就返回 NaN

    # 如果只有单个 tilt angle，复制成每帧
    tilt_angles = np.full(frames.shape[0], float(tilt_angle), dtype=np.float32)

    return frames, tilt_angles
