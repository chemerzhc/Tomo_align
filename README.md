# Optimization based tomography alignment method
A python based algorithm for electron/X-ray tomography dataset fin alignment
# Tomo_alignment

<img width="803" height="782" alt="image" src="https://github.com/user-attachments/assets/b5188301-098e-43b2-8c64-c119c542b712" />

**A modular tomography alignment pipeline for electron/X-ray tomography, including:**

- Preprocessing of raw EMD tomography series
- Global alignment (centroid + cross-correlation)
- Common-line alignment refinement
- Projection consistency refinement
- GPU acceleration (CuPy) support
- YAML-based reproducible experiment configuration

---

## Features

- Modular multi-stage tomography pipeline
- Config-driven experiments via YAML
- GPU / CPU backend support
- Reproducible experiment logging
- Automatic config snapshot saving
- Easy stage on/off control

---

# Installation

## 1. Clone repository

On a new machine:

```bash
git clone https://github.com/chemerzhc/tomo_alignment.git
cd tomo_alignment
```

---

## 2. Recommended Installation (Conda)

Recommended for reproducibility and GPU support.

Create environment:

```bash
conda env create -f environment.yml
```

Activate environment:

### Windows

```bash
conda activate tomo_TV
```

### Linux / Mac

```bash
conda activate tomo_TV
```

Install package in editable mode:

```bash
pip install -e .
```

Verify installation:

```bash
python -c "import tomography; print('tomography installed successfully')"
```

If successful, no error should appear.

---

## 3. Lightweight Installation (pip)

For lightweight CPU-only installation.

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package:

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy scipy matplotlib h5py pyyaml loguru tqdm opencv-python
pip install -e .
```

Verify installation:

```bash
python -c "import tomography; print('tomography installed successfully')"
```

---

## 4. GPU Support (Optional)

The pipeline supports GPU acceleration via CuPy.

First check CUDA version:

```bash
nvidia-smi
```

Example output:

```text
CUDA Version: 12.2
```

Then install the corresponding CuPy version.

### CUDA 12.x

```bash
pip install cupy-cuda12x
```

### CUDA 11.x

```bash
pip install cupy-cuda11x
```

Replace according to your CUDA version.

Verify GPU detection:

```bash
python -c "import tomography"
```

If GPU is detected:

```text
[env] CuPy detected -> GPU enabled
```

Otherwise:

```text
[env] CuPy not found -> CPU mode
```

To force CPU mode:

Edit config:

```yaml
gpu:
  use_gpu: false
```

---

# Quick Start

After installation, run a quick experiment to verify everything works.

From the project root directory:

```bash
python run.py --params_path tomography/params/default.yml
```

If successful, output files will appear in:

```text
output/Xray/
```

Example:

```text
output/Xray/
│
├── pipeline.log
├── used_config.yml
├── test.h5
├── test.mp4
├── test.png
└── final_movie_aligned.mp4
```

If `pipeline.log` and `used_config.yml` are generated, installation is working correctly.

---

## Common Installation Errors

### ModuleNotFoundError: No module named 'tomography'

Usually caused by missing package installation.

Run:

```bash
pip install -e .
```

Then verify:

```bash
python -c "import tomography"
```

---

### Wrong Working Directory

Incorrect:

```bash
cd tomography
python run.py
```

Correct:

Run from project root:

```bash
python run.py --params_path tomography/params/default.yml
```

---

### CuPy not found

GPU acceleration unavailable.

Install CuPy:

```bash
pip install cupy-cuda12x
```

Or disable GPU:

```yaml
gpu:
  use_gpu: false
```

### Reproducibility

Each run automatically saves:

```text
used_config.yml
```

inside output directory.

This guarantees exact experiment reproducibility.

---

# Citation

[1] Odstrčil, M., Holler, M., Raabe, J., & Guizar-Sicairos, M. (2019). Alignment methods for nanotomography with deep subpixel accuracy. Optics Express, 27(25), 36637-36652.

[2] Schwartz J, Harris C, Pietryga J, Zheng H, Kumar P, Visheratina A, Kotov NA, Major B, Avery P, Ercius P, Ayachit U. Real-time 3D analysis during electron tomography using tomviz. Nature Communications. 2022 Aug 1;13(1):4458.

[3] Schwartz, J., Harris, C., Pietryga, J., Zheng, H., Kumar, P., Visheratina, A., ... & Hovden, R. (2022). Real-time 3D analysis during electron tomography using tomviz. Nature Communications, 13(1), 4458.
