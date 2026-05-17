import h5py
import numpy as np


def write_tomo_hdf5(
    output_file,
    tilt_angles,
    tilt_series,
    attrs=None
):
    """
    Write tomography tilt series and angles to an HDF5 file.

    Parameters
    ----------
    output_file : str or Path
        Output HDF5 file path.
    tilt_angles : array-like
        Tilt angles in degrees.
    tilt_series : array-like
        Tomography image stack.
    attrs : dict, optional
        Additional HDF5 file attributes.

    Returns
    -------
    None
    """
    with h5py.File(output_file, "w") as f:
        f.create_dataset(
            "tiltAngles",
            data=np.asarray(tilt_angles),
            dtype="f4"
        )

        f.create_dataset(
            "tiltSeries",
            data=np.asarray(tilt_series),
            dtype="f4"
        )

        if attrs is not None:
            for k, v in attrs.items():
                f.attrs[k] = v