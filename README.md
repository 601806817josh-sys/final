# Semi-Supervised 3D Medical Image Segmentation for CT Images

## Project overview
This repository contains a simplified academic codebase for a graduation thesis on semi-supervised 3D medical image segmentation for CT volumes.  
The project focuses on a reproducible baseline built around a **3D U-Net** and a **Mean Teacher** training strategy.

The repository intentionally keeps only what is required for:
- model development,
- training experiments,
- dataset loading,
- and evaluation.

Large research benchmark artifacts, historical scripts, and legacy prototypes were removed to keep the project concise and maintainable.

## Model architecture (3D U-Net + Mean Teacher)
The framework uses:
- **Student model**: 3D U-Net for voxel-wise segmentation.
- **Teacher model**: Exponential Moving Average (EMA) of student weights.
- **Supervised objective**: Cross-entropy + Dice loss on labeled samples.
- **Consistency objective**: Mean-squared consistency between student and teacher predictions on unlabeled samples.

This setup enables training with a small labeled subset while leveraging additional unlabeled CT scans.

## Repository structure
```text
/project
    /models        # 3D U-Net and model factory
    /datasets      # Dataset loader and transforms
    /training      # Supervised and Mean Teacher training scripts
    /utils         # Losses, ramp scheduling, and 3D evaluation helpers
    /configs       # Example YAML configs for experiments and dataset layout
README.md
requirements.txt
```

## Environment setup
1. Create and activate a Python environment (recommended Python 3.9+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training instructions
### 1) Semi-supervised Mean Teacher training
```bash
python -m project.training.train_mean_teacher_3d \
  --root_path ./data/BraTS2019 \
  --labeled_num 25 \
  --batch_size 4 \
  --labeled_bs 2
```

### 2) Fully supervised baseline
```bash
python -m project.training.train_supervised_3d \
  --root_path ./data/BraTS2019 \
  --batch_size 2
```

### Dataset notes
- Raw datasets are **not** included in this repository.
- Place only preprocessed files locally (outside version control), following the template in `project/configs/dataset_template.yaml`.

## Experiment notes
- Start with the default config in `project/configs/train_mean_teacher_3d.yaml`.
- Keep a fixed random seed when comparing runs.
- Log outputs under `runs/` and record:
  - labeled/unlabeled split,
  - patch size,
  - consistency weight and ramp-up,
  - final validation Dice/HD95.
- For thesis reporting, compare semi-supervised runs against the supervised baseline under identical preprocessing and data splits.
