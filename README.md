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

- **2. Patch Extraction:** Fixed-size 3D patches are sampled from the volume. Positive patches are centered on known aneurysm locations, while negative patches are drawn from anatomically relevant but non-aneurysmal regions. This step ensures spatial grounding and enables patch-level supervision.

- **3. Patch-Level Classification:** Each 3D patch is passed through a CNN-based classifier that predicts whether the region contains aneurysmal tissue. Beyond binary classification, the model also estimates the **vascular territory** associated with a positive patch. To support this spatial reasoning, the classifier incorporates auxiliary inputs such as the **world-coordinate center** of the patch and the **scan modality** (e.g., CTA, MRA), allowing it to contextualize each patch within the cerebrovascular anatomy.

## 🧠 Model Architecture

The patch classifier combines a pretrained 3D convolutional backbone with lightweight metadata embeddings to produce aneurysm-related predictions:

**3D Patch → MedicalNet 3D CNN → Feature Fusion (modality + coordinates) → Classification Head**

Key features:

- Uses a [**MedicalNet 3D ResNet**](https://github.com/Warvito/MedicalNet-models) backbone pretrained on large medical-imaging corpora for robust volumetric feature extraction.
- Incorporates **world-coordinate embeddings** and **modality embeddings** to provide anatomical and acquisition context.
- Produces a **multi-class prediction**, covering aneurysm presence as well as the vascular territory associated with a positive patch.

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
