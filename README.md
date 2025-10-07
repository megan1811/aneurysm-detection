# 🧠 Aneurysm Detection

kaggle competition & dataset: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview

This project explores automated detection of intracranial aneurysms using multi-modal brain imaging data (CT, CTA, and MRA). The goal is to build an explainable model capable of identifying aneurysm presence and location from 3D series, segmentation masks, and bounding box annotations.

# 📊 Data Analysis Overview

Before modeling, an extensive exploratory analysis was conducted to validate dataset integrity and guide design choices.
We verified unique IDs across all data sources (train.csv, train_localizers.csv, segmentation folders), ensured label consistency, and visualized modality distribution, label prevalence, and co-occurrence patterns. Bounding box coordinates were analyzed to confirm alignment with multi-hot labels and assess spatial patterns across modalities.
These findings shaped preprocessing pipelines and informed which subsets were used for model training.
