import h5py
import numpy as np


def write_tomo_hdf5(
    output_file,
    tilt_angles,
    tilt_series,
    attrs=None
):
    with h5py.File(output_file, "w") as f:
        f.create_dataset(
            "tiltAngles", data=np.asarray(tilt_angles), dtype="f4"
        )
        f.create_dataset(
            "tiltSeries", data=np.asarray(tilt_series), dtype="f4"
        )

        if attrs is not None:
            for k, v in attrs.items():
                f.attrs[k] = v
