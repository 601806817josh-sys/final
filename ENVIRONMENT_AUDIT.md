# Environment Audit (SSL4MIS)

## Runtime detection
- Python executable: `/root/.pyenv/versions/3.10.19/bin/python3`
- Python version: `3.10.19`
- `sys.prefix == sys.base_prefix` and both `VIRTUAL_ENV`/`CONDA_PREFIX` are unset.
- Interpretation: **no active virtual environment**; running on the local/system pyenv interpreter.

## Dependency sources inspected
- `README.md`
- `requirements.txt`
- `project/training/train_mean_teacher_3d.py`
- `project/training/train_supervised_3d.py`
- `project/models/*.py`
- `project/datasets/brats2019.py`
- `project/utils/*.py`
- `environment.yml` was checked and is **not present** in this repository.

## Required dependencies inferred
From `requirements.txt`:
- torch
- torchvision
- numpy
- h5py
- tensorboardX
- tqdm
- medpy
- SimpleITK
- nibabel
- PyYAML

From code imports, the external modules used directly are:
- torch
- torchvision
- numpy
- h5py
- tensorboardX
- tqdm
- medpy

## Current installation status
- Installed: *(none of the required project packages were importable)*
- Missing:
  - torch
  - torchvision
  - numpy
  - h5py
  - tensorboardX
  - tqdm
  - medpy
  - SimpleITK
  - nibabel
  - PyYAML

## Installation attempts
Attempted with:
- `python3 -m pip install -r requirements.txt`
- `env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python3 -m pip install -r requirements.txt`

Both failed before download due network/proxy connectivity to package index. As a result, required dependencies could not be installed in this environment.

## Readiness
- Project is **not ready to run training** until dependencies are installable.
- First training entry script (once dependencies are installed):
  - `python -m project.training.train_mean_teacher_3d --root_path ./data/BraTS2019 --labeled_num 25 --batch_size 4 --labeled_bs 2`
