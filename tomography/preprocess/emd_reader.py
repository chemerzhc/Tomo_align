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
    frames : np.ndarray of shape (n_frames, Y, X)
        Image stack.
    tilt_angles : np.ndarray of shape (n_frames,)
        Tilt angles in degrees. Replicated if only one value exists.
    """
    if not emd_file.exists():
        raise FileNotFoundError(f"File does not exist: {emd_file}")

    try:
        s = hs.load(str(emd_file), load_original_metadata=True, lazy=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load EMD file {emd_file}: {e}")

    data = np.asarray(s.data)

    if data.ndim == 2:
        frames = data[np.newaxis, :, :].astype(np.float32)
    elif data.ndim == 3:
        frames = data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected data shape {data.shape} in {emd_file}")

    try:
        tilt_angle = s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
    except AttributeError:
        tilt_angle = np.nan

    tilt_angles = np.full(frames.shape[0], float(tilt_angle), dtype=np.float32)

    return frames, tilt_angles