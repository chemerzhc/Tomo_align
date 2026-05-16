# Tomography optimization based alignment method
A python based algorithm for electron/X-ray tomography dataset fin alignment
# tomo_alignment

<img width="803" height="782" alt="image" src="https://github.com/user-attachments/assets/b5188301-098e-43b2-8c64-c119c542b712" />

A modular tomography alignment pipeline for electron/X-ray tomography, including:

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

## Repository Structure

```text
tomo_alignment/
│
├── run.py
│
├── tomography/
│   ├── alignment/
│   ├── preprocess/
│   ├── reconstruction/
│   ├── utils/
│   │   ├── config.py
│   │   └── logger.py
│   │
│   ├── params/
│   │   └── default.yml
│   │
│   └── __init__.py
│
├── output/
│
└── README.md
```

---

## Installation

## 1. Clone repository

On a new machine:

```bash
git clone https://github.com/YOUR_USERNAME/tomo_alignment.git
```

Move into the project directory:

```bash
cd tomo_alignment
```

---

## 2. Create conda environment

Recommended:

```bash
conda create -n tomo_TV python=3.10
```

Activate environment:

### Windows

```bash
conda activate tomo_TV
```

### Linux / Mac

```bash
source activate tomo_TV
```

---

## 3. Install dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install numpy scipy matplotlib h5py pyyaml loguru tqdm opencv-python
```

For GPU acceleration:

```bash
pip install cupy-cuda12x
```

Replace CUDA version accordingly.

Example:

- CUDA 11.x → `cupy-cuda11x`
- CUDA 12.x → `cupy-cuda12x`

---

## GPU Support

The pipeline automatically detects CuPy.

If CuPy is installed:

```text
[env] CuPy detected -> GPU enabled
```

Otherwise:

```text
[env] CuPy not found -> CPU mode
```

To force CPU mode:

Edit:

```yaml
gpu:
  use_gpu: false
```

---

## Configuration

All experiments are controlled through YAML files.

Configs are stored under:

```text
tomography/params/
```

Example:

```text
tomography/params/default.yml
```

---

## Example configuration

```yaml
project:
  output_dir: "output/Xray"

gpu:
  use_gpu: true

steps:
  preprocess: true
  global_align: true
  common_line: true
  pc_refine: true

preprocess:
  emd_dir: "data/Tomo_series"
  output_h5: "output/Xray/output.h5"

global_alignment:

  h5_file: "output/Xray/output.h5"

  output_file: "output/Xray/test.h5"
  movie_file: "output/Xray/test.mp4"
  montage_path: "output/Xray/test.png"

  max_iter_centroid: 100
  lr_centroid: 0.3
  tol_centroid: 0.01

  max_iter_cc: 100
  lr_cc: 0.5
  tol_cc: 0.05

common_line:

  h5_file: "output/Xray/test.h5"

  max_iter: 20
  lr: 0.7

  output_file:
    "output/Xray/comline.h5"

pc_refinement:

  aligned_h5_file:
    "output/Xray/test.h5"

  output_file:
    "output/Xray/refined.h5"

  Nx_target: 224
  Ny_target: 224
  Nz_target: 224
```

---

# Running the Pipeline

Run from the **project root directory**.

Example:

```bash
python run.py --params_path tomography/params/default.yml
```

Example for a custom experiment:

```bash
python run.py --params_path tomography/params/ruo2.yml
```

---

# Pipeline Stages

Each stage can be enabled or disabled.

Inside YAML:

```yaml
steps:
  preprocess: true
  global_align: true
  common_line: true
  pc_refine: true
```

Example:

Skip preprocessing:

```yaml
steps:
  preprocess: false
```

This allows restarting from intermediate outputs.

---

# Output Files

All outputs are saved to:

```text
output/
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

### pipeline.log

Contains runtime logs.

### used_config.yml

Snapshot of the exact config used.

Useful for reproducibility.

---

# Example Workflow

## Step 1

Prepare tomography series:

```text
data/
└── Tomo_series/
```

---

## Step 2

Edit config:

```text
tomography/params/default.yml
```

Set:

```yaml
preprocess:
  emd_dir: "data/Tomo_series"
```

---

## Step 3

Run pipeline:

```bash
python run.py --params_path tomography/params/default.yml
```

---

## Step 4

Check outputs:

```text
output/Xray/
```

---

# Common Errors

## ModuleNotFoundError: No module named 'tomography'

Wrong working directory.

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

## CuPy not found

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

---

## FileNotFoundError for config

Ensure config exists:

```text
tomography/params/default.yml
```

And run:

```bash
python run.py --params_path tomography/params/default.yml
```

---

### Reproducibility

Each run automatically saves:

```text
used_config.yml
```

inside output directory.

This guarantees exact experiment reproducibility.

---

# Citation

Work in progress.
