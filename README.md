# 🧠 Aneurysm Detection

## 📘 Introduction

This repository implements a 3D medical-imaging pipeline for detecting intracranial aneurysms from CT/MRI brain scans. The current stage focuses on **patch-level classification**, in which localized 3D patches are sampled from the volume and used to train a classifier that distinguishes **aneurysm** from **non-aneurysm** tissue.

The project addresses key challenges in medical computer vision, including heterogeneous scanning protocols, spatially grounded preprocessing, and severe class imbalance. The codebase is structured for clarity and extensibility, enabling future progression toward **scan-level detection** and **aneurysm localization**.

⚠️ This project is under development.

## 🩻 Dataset

This project uses the **RSNA Intracranial Aneurysm Detection** dataset, a large multi-institutional collection of brain imaging studies. It includes several thousand 3D series across multiple modalities — primarily **CT angiography (CTA)**, **MRA**, and post-contrast or T2-weighted **MRI**. All scans are provided in **DICOM** format. Expert neuroradiologists annotated each study for aneurysm presence and, when applicable, assigned each aneurysm to one of **13 predefined vascular territories**.

The dataset is highly heterogeneous in scanner type, acquisition protocol, and voxel spacing, making it a realistic and technically challenging benchmark for 3D medical computer vision.

For more details, see the official dataset page:  
https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview

## 🧬 Pipeline Summary (Patch Classification)

The goal of this pipeline is to create a reliable workflow for turning raw cerebrovascular scans into labeled 3D patches suitable for aneurysm classification. The process consists of three main stages:

- **1. Volume Loading & Preprocessing:** Raw DICOM series are converted into 3D volumes using SimpleITK. Each scan is resampled to an isotropic voxel spacing and intensity-normalized to reduce variability across scanners and acquisition protocols.

- **2. Patch Extraction:** Fixed-size 3D patches are sampled from each volume. Positive patches are centered on annotated aneurysm locations, while negative patches are drawn from non-aneurysmal regions. This step ensures spatial grounding and enables patch-level supervision.

- **3. Patch-Level Classification:** Each patch is processed by a 3D CNN that predicts (1) whether an aneurysm is present and, if so, (2) the associated intracranial vascular territory. In addition to image data, the model consumes auxiliary metadata — the **normalized patch center coordinates**(scaled relative to the scan volume) and the **scan modality** — to provide spatial and acquisition context.

## 🧠 Model Architecture

The patch classifier is a two-head model built on top of a pretrained 3D convolutional backbone:

**3D Patch → MedicalNet 3D CNN → Feature Fusion → Presence Head + Location Head**

Key Components:

- A [**MedicalNet 3D ResNet**](https://github.com/Warvito/MedicalNet-models)
  backbone extracts volumetric features from each patch.
- Patch features are fused with learned embeddings of the **scan modality**, and the normalized **3D patch center coordinates**.
- The fused representation feeds two prediction heads (1) a **binary presence head** (aneurysm vs. no aneurysm) and (2) a **multiclass location head** predicting one of 13 vascular territories.

During training, the location head is optimized only on positive patches via masking, while the presence head is trained on all samples. The relative contribution of the location loss is controlled by a weighting factor (alpha_loc).

## 📊 Metrics

For the moment, all metrics are computed at the patch level and reflect the two prediction tasks:

- **Presence:** AUROC over all patches, chosen to handle strong class imbalance.
- **Location:** Accuracy and balanced accuracy over the 13 vascular territories, computed only on positive patches.

## 📂 Project Structure

```bash
├── src/                               # Core source code
│   ├── utils/
│   │   ├── classifier_training_functions.py   # Training utilities (loops, schedulers, helpers)
│   │   ├── CONSTANTS.py                       # Global constants (paths, mappings, config values)
│   │   ├── datasets.py                        # Dataset classes for loading patches & metadata
│   │   ├── models.py                          # Patch classifier (MedicalNet backbone + embeddings)
│   │   ├── preprocess.py                      # DICOM loading, resampling, normalization
│   ├── classifier_train_and_evaluate.py       # Runs the full training/evaluation pipeline
│   ├── dataset_creation.py                    # Generates dataset metadata: world coords, labels, modality info, splits
│   ├── patch_creation.py                      # Loads DICOM series, preprocesses volumes, extracts 3D patches
│
├── data-analysis.ipynb                        # Exploratory analysis and QA
├── visualize.ipynb                            # Visualizations of volumes, patches, and coordinates
│
├── pyproject.toml                             # Poetry environment configuration
├── poetry.lock                                # Locked dependency versions
└── README.md                                  # Project documentation
```

## 🐾 Next Steps

The current pipeline performs **patch-level** aneurysm detection. The next stage is to extend the system to the **scan level**, where predictions across all patches in a volume are aggregated into a study-level classification across the 13 intracranial vascular territories.

This will involve:

- running the patch classifier over all patches extracted from a scan,
- designing an aggregation method to combine patch-level outputs into a single scan-level prediction.

Future extensions may include exploring different aggregation mechanisms (e.g., attention-based pooling or transformer-style aggregators), integrating vessel segmentations to improve sampling strategies, and benchmarking alternative 3D backbones.
