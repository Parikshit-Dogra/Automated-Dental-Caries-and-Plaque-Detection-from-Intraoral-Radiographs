# 🦷 DentalVision AI: Automated Dental Caries and Plaque Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-UNet%20(ResNet--34)-darkgreen.svg)](https://github.com/qubvel/segmentation_models.pytorch)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Clinical%20AI-Research%20%26%20Diagnostic%20Suite-purple.svg)](#)

An end-to-end medical deep learning diagnostic system for automated semantic segmentation, localization, and quantitative assessment of **Dental Caries (tooth decay)** from intraoral periapical radiographs and **Bacterial Dental Plaque (biofilm)** from intraoral clinical photographs.

The project pairs deep convolutional networks (UNet with pre-trained ResNet-34 encoders) with an interactive **Streamlit Clinical Decision Support Suite**, real-time **CLAHE contrast enhancement**, connected-component lesion quantification, and a curated **benchmark testing suite** with radiologist-verified ground-truth references.

---

## 📋 Table of Contents
- [Key Features](#-key-features)
- [Clinical Modalities & Diagnostic Capabilities](#-clinical-modalities--diagnostic-capabilities)
- [Dataset Sources & References](#-dataset-sources--references)
- [System Architecture & Methodology](#-system-architecture--methodology)
- [Model Evaluation & Benchmarks](#-model-evaluation--benchmarks)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Streamlit Clinical Diagnostic Suite](#-streamlit-clinical-diagnostic-suite)
- [Curated Demo Testing Suite](#-curated-demo-testing-suite)
- [Model Training & Notebooks](#-model-training--notebooks)
- [Clinical Severity Grading Scheme](#-clinical-severity-grading-scheme)
- [Tech Stack](#-tech-stack)
- [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🌟 Key Features

- **Dual Clinical Modality Diagnostics**:
  - **Intraoral Radiographs**: Detects and segments subtle coronal and interproximal carious lesions, demineralization, and recurrent cavities adjacent to restorations.
  - **Intraoral Clinical Photographs**: Segments oral biofilm and dental plaque deposits across enamel surfaces and cervical gingival margins.
- **Deep Semantic Segmentation (UNet + ResNet-34)**:
  - Transfer learning leveraging ImageNet pre-trained ResNet-34 encoders with skip connections for boundary delineation.
  - Trained using a compound **BCE + Soft Dice Loss** to handle extreme positive-to-background class imbalance.
- **Adaptive Image Preprocessing**:
  - Integrated **Contrast Limited Adaptive Histogram Equalization (CLAHE)** for low-contrast dental X-rays to highlight subtle radiolucencies without amplifying noise.
- **Automated Quantification & Localization**:
  - Connected component analysis identifying lesion count, centroid coordinates $(X, Y)$, bounding boxes, pixel surface area, and overall percentage of crown/image coverage.
- **Automated Severity Classification**:
  - Rule-based clinical triage categorizing findings from sound/caries-free to incipient, moderate, or severe multi-surface pathology.
- **Interactive Web Diagnostic Suite**:
  - Streamlit dashboard providing real-time inference, threshold sensitivity tuning, opacity sliders, bounding box toggles, localized metric tables, and side-by-side ground truth verification.
- **Curated Clinical Benchmark Suite**:
  - Dedicated test set with expected diagnostic findings and radiologist ground-truth visual references for sound teeth, incipient decay, and multi-surface caries.

---

## 🩺 Clinical Modalities & Diagnostic Capabilities

| Modality | Target Pathology | Imaging Type | Key Clinical Challenge | Preprocessing Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Mode 1: Caries Detection** | Dental Caries (Demineralized Cavities) | Intraoral Periapical Radiographs (X-ray) | Low contrast, radio-opaque restorations, subtle interproximal demineralization | Grayscale conversion, CLAHE $(2.0 \text{ clip limit}, 8\times8 \text{ tiles})$, ImageNet normalization |
| **Mode 2: Plaque Detection** | Bacterial Plaque & Biofilm Deposits | Intraoral Clinical Photographic Images | Complex oral illumination, gingival reflection, variable biofilm thickness | Full-color RGB processing, ImageNet 3-channel normalization |

---

## 📚 Dataset Sources & References

This project leverages two publicly available clinical dental datasets:

### 1. Intraoral Radiographic Database for Periapical Lesions & Caries (Mendeley Data)
- **Source Link**: [Mendeley Data Repository (Dataset ID: 4fbdxs7s7w/1)](https://data.mendeley.com/datasets/4fbdxs7s7w/1)
- **Direct URL**: https://data.mendeley.com/datasets/4fbdxs7s7w/1
- **Imaging Modality**: Intraoral Periapical Radiographs (X-rays)
- **Format & Ground Truth**: JPEG radiographs with associated YOLO-format normalized polygonal coordinate labels (`classes.txt`: `Caries`, `Lesion_Periapical`, `Restoration`).
- **Project Role**: Used in [Notebooks/Caries.ipynb](file:///d:/GItRepo/Automated%20Dental%20Caries%20and%20Plaque%20Detection%20from%20Intraoral%20Radiographs/Notebooks/Caries.ipynb) to train and evaluate the Caries UNet network (`models/best_caries_unet.pth`) for localized detection of interproximal and coronal carious lesions.

### 2. Oral Disease Segmentation (ODS) Dataset (Oralformer)
- **Source Link**: [Oralformer GitHub Repository (LintaoPeng/Oralformer)](https://github.com/LintaoPeng/Oralformer)
- **Direct URL**: https://github.com/LintaoPeng/Oralformer
- **Imaging Modality**: High-resolution intraoral clinical color photographs (RGB)
- **Format & Ground Truth**: Paired RGB photographs and pixel-level binary ground-truth segmentation masks across three distinct oral conditions:
  - `Plaques`: Microbial biofilm deposited on tooth surfaces and cervical gingival margins.
  - `Caries`: Clinical photo-visible cavitated lesions.
  - `Calculus`: Supra- and subgingival calcified deposits.
- **Project Role**: Used in [Notebooks/Plaque.ipynb](file:///d:/GItRepo/Automated%20Dental%20Caries%20and%20Plaque%20Detection%20from%20Intraoral%20Radiographs/Notebooks/Plaque.ipynb) to train and evaluate the Plaque UNet network (`models/plaque_unet.pth`) for automated plaque biofilm quantification and oral hygiene grading.

---

## 🏗️ System Architecture & Methodology

```
                          [ Input Dental Image ]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
   [ Intraoral Radiograph ]                   [ Clinical Photograph ]
               │                                         │
       [ CLAHE Contrast ]                       [ Color Preserved ]
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         [ ImageNet Normalization ]
                         [ Resize to 512 x 512 ]
                                    │
                                    ▼
                     [ UNet Feature Extraction ]
                     - Encoder: Pretrained ResNet-34
                     - Skip Connections to Decoder
                     - Sigmoid Probability Output
                                    │
                                    ▼
                     [ Decision Thresholding ]
                     - Adjustable Sensitivity (0.10 - 0.90)
                                    │
                                    ▼
                    [ Contour & Morphological Analysis ]
                     - Min Area Filter (Noise Rejection)
                     - Centroid & Bounding Box Calculation
                     - Affected Area % Coverage Computation
                                    │
                                    ▼
                    [ Clinical Diagnostic Report ]
                     - Visual Segmentation Mask & Contours
                     - Quantitative Findings Data Table
                     - Severity Triage Assessment
```

### 1. Model Architecture
- **Backbone**: UNet with ResNet-34 encoder (`segmentation_models_pytorch`).
- **Input Dimensions**: $512 \times 512 \times 3$.
- **Activation**: Differentiable Sigmoid generating pixel-wise probability maps $P(y_{i,j}=1|x)$.

### 2. Loss Function Formulation
To resolve extreme class imbalance (carious lesions and plaque typically occupy $<5\%$ of total image area), models were trained using a balanced compound loss:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$$

$$\mathcal{L}_{\text{SoftDice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$

Where $p_i$ is the predicted probability and $g_i \in \{0, 1\}$ is the ground-truth binary label.

---

## 📊 Model Evaluation & Benchmarks

Models were validated against unseen clinical test splits evaluating Dice Similarity Coefficient (DSC), Sensitivity (Recall), Specificity, Precision, and Intersection over Union (IoU / Jaccard).

### 1. Dental Caries UNet (Intraoral Radiographs)
- **Dataset**: [Intraoral Radiographic Database for Periapical Lesions & Caries (Mendeley Data)](https://data.mendeley.com/datasets/4fbdxs7s7w/1).
- **Best Validation Dice**: `0.5515` (Epoch 37/40).
- **Test Performance across Decision Thresholds**:

| Threshold | Dice (DSC) | Sensitivity (Recall) | Specificity | Precision | IoU (Jaccard) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | 0.4157 | 0.3517 | 0.9981 | 0.5081 | 0.2623 |
| **0.40** *(App Default)* | **0.4157** | **0.3487** | **0.9981** | **0.5146** | **0.2624** |
| **0.50** | 0.4159 | 0.3462 | 0.9982 | 0.5207 | 0.2625 |
| **0.60** *(Max Dice)* | **0.4160** | **0.3436** | **0.9983** | **0.5272** | **0.2627** |

> **Clinical Insight**: The Caries UNet achieves high specificity ($>99.8\%$), minimizing false positives on healthy teeth and existing restorations, while localizing deep dentinal and coronal caries.

---

### 2. Dental Plaque UNet (Clinical Photographs)
- **Dataset**: [Oral Disease Segmentation (ODS) Dataset (Oralformer)](https://github.com/LintaoPeng/Oralformer) (Plaques subset).
- **Peak Validation Dice**: `0.8159` (Epoch 40/40).
- **Test Performance across Decision Thresholds**:

| Threshold | Dice (DSC) | Sensitivity (Recall) | Specificity | Precision | IoU (Jaccard) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.35** | 0.8394 | 0.8542 | 0.9842 | 0.8251 | 0.7233 |
| **0.40** *(App Default)* | **0.8399** | **0.8508** | **0.9847** | **0.8293** | **0.7240** |
| **0.50** | 0.8402 | 0.8435 | 0.9857 | 0.8369 | 0.7244 |
| **0.60** *(Optimal IoU)* | **0.8405** | **0.8357** | **0.9867** | **0.8454** | **0.7249** |

> **Clinical Insight**: Achieves **$84.05\%$ Dice similarity** and **$72.49\%$ IoU** on photographic tooth surfaces, accurately tracking both thin cervical deposits and extensive confluent biofilm.

---

## 📁 Repository Structure

```
Automated Dental Caries and Plaque Detection from Intraoral Radiographs/
├── app/
│   └── app.py                             # Streamlit diagnostic web application
│
├── Datasets/
│   ├── Intraoral Radiographic Database.../ # X-ray images, YOLO polygon labels, classes.txt
│   └── ODS_dataset/                       # Oral Disease Segmentation (Plaques, Caries, Calculus)
│
├── demo testing/                          # Structured benchmark evaluation suite
│   ├── README.md                          # Testing protocol and user guide
│   ├── caries/
│   │   ├── images/                        # 9 raw X-ray test cases (sound, mild, severe)
│   │   ├── ground_truth_reference/        # Side-by-side radiologist ground-truth references
│   │   └── EXPECTED_RESULTS.md            # Detailed diagnostic finding expectations
│   └── plaque/
│       ├── images/                        # 8 clinical photographic test cases (clean, mild, heavy)
│       ├── ground_truth_reference/        # Side-by-side photographic clinical references
│       └── EXPECTED_RESULTS.md            # Expected biofilm percentage and region counts
│
├── models/
│   ├── best_caries_unet.pth               # Trained PyTorch weights for Caries UNet (ResNet-34)
│   └── plaque_unet.pth                    # Trained PyTorch weights for Plaque UNet (ResNet-34)
│
├── Notebooks/
│   ├── Caries.ipynb                       # Dataset preparation, training loop, threshold sweep
│   └── Plaque.ipynb                       # Plaque pipeline, RAM caching, CosineAnnealing training
│
├── requirements.txt                       # Project Python dependencies
└── README.md                              # Master project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10 or 3.11
- CUDA-compatible GPU recommended (CPU execution is fully supported)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/automated-dental-caries-plaque-detection.git
cd "Automated Dental Caries and Plaque Detection from Intraoral Radiographs"
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Streamlit Clinical Diagnostic Suite

Launch the interactive diagnostic interface:

```bash
streamlit run app/app.py
```

The web dashboard opens automatically at `http://localhost:8501`.

### User Interface Guide:
1. **Sidebar Modality Selection**: Switch between:
   - `🦷 Intraoral Radiograph (Caries / Decay)`
   - `📷 Clinical Camera (Plaque / Biofilm)`
2. **Diagnostic Parameter Controls**:
   - **Sensitivity / Detection Threshold** slider (`0.10` - `0.90`, default `0.40`).
   - **CLAHE Contrast Enhancement** toggle (enhances subtle demineralization boundaries on X-rays).
   - **Diagnostic Overlay Opacity** slider.
   - **Show Bounding Boxes** toggle.
3. **Image Input Methods**:
   - **🎯 Curated Demo Testing Suite**: Select pre-loaded cases with built-in ground-truth metadata.
   - **📁 Upload Image File**: Upload any local `.jpg` or `.png` dental image.
   - **🗂️ Browse Full Dataset**: Explore clinical samples directly from the dataset directories.
4. **Diagnostic Findings Display**:
   - 4-metric header: Lesion count, affected area %, maximum probability, and severity badge.
   - 3-column visual comparison: Raw image, preprocessed image, and diagnostic overlay.
   - Detailed quantitative findings table with centroid coordinates and bounding box dimensions.
   - Expandable side-by-side ground-truth viewer for validation.

---

## 🧪 Curated Demo Testing Suite

For systematic validation, use the curated cases in `demo testing/`:

### Selected Caries Test Cases (`demo testing/caries/`)
| Image | Pathology Status | Expected Lesions | Area % | Key Landmark |
| :--- | :--- | :---: | :---: | :--- |
| `patient_0001.jpg` | Sound (Negative) | **0** | `0.00%` | Healthy enamel; 3 radio-opaque fillings present. |
| `patient_0002.jpg` | Sound (Negative) | **0** | `0.00%` | Caries-free crowns; root apex periapical lesion. |
| `patient_0004.jpg` | Single Cavity (Mild) | **1** | `2.16%` | Coronal interproximal radiolucency. |
| `patient_0016.jpg` | Multiple Cavities (Moderate) | **2** | `3.88%` | Interproximal decay across adjacent teeth. |
| `patient_0044.jpg` | Severe Multi-surface Decay | **3** | `3.44%` | Extensive coronal destruction. |

### Selected Plaque Test Cases (`demo testing/plaque/`)
| Image | Biofilm Stage | Expected Regions | Area % | Key Clinical Landmark |
| :--- | :--- | :---: | :---: | :--- |
| `10.png` | Clean Enamel | **0** | `0.00%` | Plaque-free smooth tooth surfaces. |
| `1194.png` | Mild Incipient Plaque | **1** | `0.61%` | Localized cervical margin accumulation. |
| `1069.png` | Moderate Plaque | **2** | `5.14%` | Interdental and labial plaque accumulation. |
| `100.png` | Heavy Biofilm | **2** | `15.01%` | Extensive continuous coating across crowns. |
| `1040.png` | Severe Biofilm | **3** | `16.71%` | Thick confluent multi-tooth plaque deposits. |

---

## 📓 Model Training & Notebooks

Full reproducible training pipelines are provided in `Notebooks/`:

### 1. `Notebooks/Caries.ipynb`
- **Data Parsing**: Converts YOLO-style polygon segmentation labels to binary raster masks.
- **Augmentation**: Albumentations pipeline including horizontal flips, vertical flips, random shifts, and scaling.
- **Training**: 40 epochs with AdamW ($\text{LR} = 10^{-4}$), Cosine Annealing learning rate schedule, and `CombinedBceDiceLoss`.
- **Threshold Optimization**: Comprehensive evaluation across decision thresholds $0.1$ to $0.9$.

### 2. `Notebooks/Plaque.ipynb`
- **RAM Caching**: In-memory caching of high-resolution photographic image-mask pairs for accelerated training.
- **Mixed Precision**: Accelerated training using `torch.cuda.amp`.
- **Validation**: Epoch-by-epoch evaluation of validation Dice and sensitivity saving the top-performing checkpoint to `models/plaque_unet.pth`.

---

## 📐 Clinical Severity Grading Scheme

### Dental Caries (Radiographs)
- **Sound / Negative** ($0$ lesions): Intact enamel with no radiolucencies detected.
- **Incipient / Enamel Caries** ($< 1.5\%$ area): Localized demineralization confined to enamel or enamel-dentin junction.
- **Moderate - Deep Dentinal Caries** ($\ge 1.5\%$ area or multiple lesions): Decay penetrating deep into dentin approaching or involving the pulp chamber.

### Dental Plaque (Photographs)
- **Clean Surfaces** ($0\%$ area): No visible bacterial biofilm; sound enamel and gingival margins.
- **Mild Plaque Accumulation** ($< 5.0\%$ area): Incipient deposits restricted to cervical lines or interproximal embrasures.
- **Moderate - Heavy Biofilm Coverage** ($\ge 5.0\%$ area): Thick continuous plaque covering significant portions of the crown surfaces requiring clinical prophylaxis.

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
- **Model Library**: [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)
- **Computer Vision & Image Processing**: [OpenCV (cv2)](https://opencv.org/), [Albumentations](https://albumentations.ai/), [Pillow](https://python-pillow.org/)
- **Clinical Dashboard & UI**: [Streamlit](https://streamlit.io/)
- **Data Science & Analytics**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Scikit-Learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/)

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT**: This software is intended strictly for research, educational, and decision-support exploration. It is not certified as a primary diagnostic medical device by the FDA, CE, or CDSCO. Deep learning segmentations must be interpreted by a licensed dentist or oral radiologist in conjunction with comprehensive clinical examination.
