# Explainable Multimodal Fusion for Mammographic Breast Cancer Detection

This repository contains the code, notebooks, and documentation for my CM3070 project:
**Explainable Multimodal Fusion of CNN and Radiomics Features for Mammographic Breast Cancer Detection Using DDSM**.

## Project Overview
This project develops an explainable multimodal diagnostic system that integrates:
- Convolutional Neural Networks (CNNs)
- Radiomics feature extraction
- Explainable AI (Grad-CAM, Grad-CAM++, Integrated Gradients)

The goal is to evaluate whether multimodal fusion improves classification performance and
explainability on the DDSM and CBIS-DDSM datasets.

## Repository Structure
multimodal-mammography-xai/
│
├── data/
│   ├── raw/                # Original dataset files (DO NOT upload DDSM images)
│   ├── processed/          # Preprocessed ROIs, masks, etc.
│   └── radiomics/          # Extracted radiomics CSVs
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_cnn_training.ipynb
│   ├── 03_radiomics_extraction.ipynb
│   ├── 04_fusion_model.ipynb
│   └── 05_explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── cnn_models.py
│   ├── radiomics_utils.py
│   ├── fusion.py
│   └── explainability.py
│
├── results/
│   ├── figures/
│   ├── heatmaps/
│   └── metrics/
│
├── docs/
│   ├── report/
│   └── diagrams/
│
├── tests/
│   └── test_preprocessing.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE


## How to Run
1. Create a virtual environment  
2. Install dependencies:  

