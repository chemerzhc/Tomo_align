# Optimization based tomography alignment method
A python based algorithm for electron/X-ray tomography dataset fine alignment
# Tomo_alignment

<img width="400" height="390" alt="image" src="https://github.com/user-attachments/assets/b5188301-098e-43b2-8c64-c119c542b712" />

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
git clone https://github.com/chemerzhc/Tomo_align.git
cd Tomo_align
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

It is recommended to first create a clean Python environment.

### Option A: Conda environment

Create environment:

```bash
conda create -n tomo_TV python=3.10
```

Activate environment:

```bash
conda activate tomo_TV
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package:

```bash
pip install -e .
```

Verify installation:

```bash
python -c "import tomography; print('tomography installed successfully')"
```

---

### Option B: Python virtual environment (venv)

Create virtual environment:

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package:

```bash
pip install -e .
```

Verify installation:

```bash
python -c "import tomography; print('tomography installed successfully')"
```

---

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
---

# Repository Structure

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

# Configuration

All experiments are controlled through YAML configuration files.

Configs are stored under:

```text
tomography/params/
```

Default example:

```text
tomography/params/default.yml
```

---

## Example Configuration

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

Default configuration:

```bash
python run.py --params_path tomography/params/default.yml
```

Custom experiment:

```bash
python run.py --params_path tomography/params/ruo2.yml
```

---

# Pipeline Stages

Each stage can be enabled or disabled independently.

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

This allows restarting from intermediate outputs without rerunning the entire pipeline.

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

Contains runtime logs for debugging and experiment tracking.

### used_config.yml

Automatically saves the exact configuration used for reproducibility.

---

# Example Workflow

## Step 1: Prepare tomography series

Prepare data under:

```text
data/
└── Tomo_series/
```

---

## Step 2: Edit configuration

Open:

```text
tomography/params/default.yml
```

Set:

```yaml
preprocess:
  emd_dir: "data/Tomo_series"
```

---

## Step 3: Run pipeline

```bash
python run.py --params_path tomography/params/default.yml
```

---

## Step 4: Check outputs

Results will be generated under:

```text
output/Xray/
```

---

# Examples

## Example 1: X-ray Tomography Alignment

### Input

(Add representative raw tomography projections here)

```text
Placeholder for input projection image / GIF
```

### Alignment Process

(Add intermediate alignment visualization here)

```text
Placeholder for alignment movie / optimization process
```

### Output

(Add reconstructed or aligned result here)

```text
Placeholder for final aligned reconstruction
```

### Example results
---

<img width="1797" height="906" alt="image" src="https://github.com/user-attachments/assets/030c4ae8-c709-4b51-8b19-2f14dcfbc9f2" />
---

# Common Errors

## ModuleNotFoundError: No module named 'tomography'

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

## Wrong Working Directory

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

### CUDA 12.x

```bash
pip install cupy-cuda12x
```

### CUDA 11.x

```bash
pip install cupy-cuda11x
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

Run:

```bash
python run.py --params_path tomography/params/default.yml
```

---

# Reproducibility

Each run automatically saves:

```text
used_config.yml
```

inside the output directory.

This guarantees exact experiment reproducibility.

---

# Citation

[1] Odstrčil, M., Holler, M., Raabe, J., & Guizar-Sicairos, M. (2019). Alignment methods for nanotomography with deep subpixel accuracy. Optics Express, 27(25), 36637-36652.

[2] Schwartz J, Harris C, Pietryga J, Zheng H, Kumar P, Visheratina A, Kotov NA, Major B, Avery P, Ercius P, Ayachit U. Real-time 3D analysis during electron tomography using tomviz. Nature Communications. 2022 Aug 1;13(1):4458.

[3] Schwartz, J., Harris, C., Pietryga, J., Zheng, H., Kumar, P., Visheratina, A., ... & Hovden, R. (2022). Real-time 3D analysis during electron tomography using tomviz. Nature Communications, 13(1), 4458.
