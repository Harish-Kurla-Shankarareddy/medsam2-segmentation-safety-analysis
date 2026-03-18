# MedSAM2 Segmentation Safety Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Medical Imaging](https://img.shields.io/badge/Medical-AI-green)

Robustness and failure analysis of **MedSAM2 medical image segmentation** using **prompt jitter experiments** on the **Medical Segmentation Decathlon (MSD) Spleen CT dataset**.

This project evaluates how sensitive MedSAM2 is to small variations in bounding box prompts and analyzes segmentation performance using multiple metrics including Dice score, false positives, false negatives, Hausdorff distance, and worst-slice Dice.

---

# Project Overview

MedSAM2 is a promptable segmentation model for 3D medical images based on the Segment Anything Model (SAM2). It allows segmentation using prompts such as bounding boxes or points.

However, prompt-based segmentation systems can be sensitive to slight changes in prompts.

This project investigates:

- robustness of MedSAM2 segmentation
- effect of bounding box perturbations
- segmentation failure modes
- worst-slice segmentation errors
- uncertainty estimation from prompt perturbations

The experiments are conducted using the **MSD Spleen CT dataset**.

---

# Experiment Pipeline

```
CT Volume
   ↓
Preprocessing (CT windowing)
   ↓
Bounding Box Prompt
   ↓
MedSAM2 Inference
   ↓
Prompt Jitter Experiments
   ↓
Segmentation Output
   ↓
Evaluation Metrics
```

Metrics evaluated:

- Dice score
- False Positive voxels
- False Negative voxels
- Hausdorff Distance (HD95)
- Worst Slice Dice

---

# Methodology

## Dataset

Experiments are performed on the **Medical Segmentation Decathlon (MSD) Spleen CT dataset**.

Dataset link:

https://medicaldecathlon.com/

Each case contains:

- a 3D abdominal CT volume
- a binary spleen segmentation mask

---

## Baseline Segmentation

Baseline MedSAM2 inference is performed on CT volumes using a bounding-box prompt derived from the ground truth mask.

Processing steps:

1. Load CT and ground truth volume
2. Apply CT windowing to normalize intensities
3. Identify the **key slice** (slice with maximum spleen area)
4. Generate bounding box prompt from the ground truth mask
5. Run MedSAM2 segmentation
6. Propagate segmentation through the 3D volume

---

## Prompt Jitter Experiment

To evaluate robustness, the bounding box prompt is randomly perturbed.

Jitter levels tested:

```
0 px
2 px
5 px
10 px
20 px
```

For each jitter level, multiple trials are performed with different random seeds.

This simulates **annotation variability in real clinical settings**.

---

## Evaluation Metrics

The following metrics are computed:

### Dice Score
Measures overlap between prediction and ground truth.

### False Positive Voxels (FP)
Voxels predicted as spleen but not present in ground truth.

### False Negative Voxels (FN)
Voxels belonging to spleen but missed by the model.

### Hausdorff Distance (HD95)
Boundary distance between predicted and ground truth segmentation.

### Worst Slice Dice
Minimum Dice score across slices of a volume.  
Used to identify **localized segmentation failures**.

---

# Uncertainty Estimation

To analyze model confidence, segmentation is repeated across multiple prompt perturbations.

For each voxel:

```
p(x) = average prediction probability across trials
```

Uncertainty is computed as:

```
U(x) = p(x)(1 − p(x))
```

Properties:

- Low uncertainty → predictions agree
- High uncertainty → predictions vary across trials

Uncertainty maps highlight:

- unstable boundaries
- ambiguous anatomy
- failure-prone regions

---

# Results

## Multi-Case Prompt Jitter Results

| Jitter (px) | Mean Dice | Mean FN Voxels | Mean FP Voxels | Mean HD95 (mm) | Worst Slice Dice |
|-------------|-----------|---------------|---------------|---------------|-----------------|
| 0           | 0.962     | 3470          | 4127          | 2.97          | 0.524           |
| 2           | 0.963     | 3403          | 4163          | 2.99          | 0.526           |
| 5           | 0.962     | 3409          | 4231          | 3.08          | 0.529           |
| 10          | 0.962     | 3603          | 4102          | 3.12          | 0.527           |
| 20          | 0.897     | 12275         | 3163          | 11.64         | 0.431           |

---

# Example Result Plots

*(Add your plots in a folder called `results/`)*

### Dice vs Prompt Jitter

```
results/dice_vs_jitter.png
```

### False Negatives vs Prompt Jitter

```
results/fn_vs_jitter.png
```

### False Positives vs Prompt Jitter

```
results/fp_vs_jitter.png
```

### HD95 vs Prompt Jitter

```
results/hd95_vs_jitter.png
```

### Worst Slice Dice vs Jitter

```
results/worst_slice_dice_vs_jitter.png
```

---

# Visualization Tool

This project includes an interactive viewer for segmentation debugging.

Run:

```
python view_case.py \
--ct path_to_ct.nii.gz \
--gt path_to_gt.nii.gz \
--pred path_to_prediction.nii.gz
```

Keyboard controls:

```
j / k → scroll slices
g → toggle ground truth
p → toggle prediction
```

The viewer displays:

- CT slice
- predicted segmentation
- ground truth
- false positive regions
- false negative regions

---

# Installation

Create environment:

```
conda create -n medsam2 python=3.10
conda activate medsam2
```

Install dependencies:

```
pip install torch torchvision
pip install numpy SimpleITK matplotlib scikit-image
```

Clone repository:

```
git clone https://github.com/Harish-Kurla-Shankarareddy/medsam2-segmentation-safety-analysis.git
cd medsam2-segmentation-safety-analysis
```

---

# Running Experiments

### Baseline inference

```
python msd_spleen_medsam2_infer.py
```

### Prompt jitter experiments

```
python msd_prompt_jitter_multicase.py
```

### Uncertainty experiments

```
python msd_multi_uncertainty_jitter.py
```

### Plot metrics

```
python plot_slice_dice.py
```

---

# Project Structure

```
medsam2-segmentation-safety-analysis

LICENSE
README.md

debug_msd_spleen.py
medsam2_infer_3D_CT.py

msd_multi_uncertainty_jitter.py
msd_prompt_jitter_multicase.py
msd_prompt_jitter_single_uncertainty.py
msd_spleen_medsam2_infer.py

plot_prompt_jitter_fp_fn.py
plot_slice_dice.py

patch_msd.py
patch_guard_import.py

resample_pred_to_native_and_dice.py
view_case.py
```

---

# Discussion

The experiments show that MedSAM2 segmentation is stable under small prompt perturbations but degrades significantly under large perturbations.

Key observations:

- Small prompt noise has minimal effect on Dice score
- Large perturbations increase **false negatives**
- Boundary errors increase with jitter
- Worst-slice Dice reveals local failures hidden by mean Dice

This highlights the importance of **robustness testing and tail-risk metrics** in medical segmentation models.

---

# Future Work

Possible extensions include:

- uncertainty-aware segmentation
- model calibration analysis
- prompt ensemble inference
- deployment with an interactive UI
- evaluation on additional MSD organs

---

# Citation

If you use MedSAM2, please cite the original work:

```
MedSAM2: Segment Anything in 3D Medical Images and Videos
Jun Ma et al.
```

---

# Author

Harish Kurla Shankarareddy  




GitHub:

https://github.com/Harish-Kurla-Shankarareddy

---

# License

Apache 2.0 License
