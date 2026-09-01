# Image Spoof Detection System

A modular face anti-spoofing system that combines **2D image forgery
detection, deepfake detection, and 3D mask spoof detection** into a
unified inference pipeline.

The system uses complementary detection approaches:

-   **MobileNetV2** for 2D image forgery detection
-   **EfficientNetB0** for deepfake / synthetic-face detection
-   **LBP + RBF-SVM** for 3D mask spoof detection through facial
    micro-texture analysis
-   A **Flask-based web application** for image upload and inference

------------------------------------------------------------------------

## Overview

Face spoofing attacks can take several forms, including digitally
manipulated images, synthetic faces, and physical presentation attacks
such as masks.

This project integrates separate specialist modules for different attack
characteristics and combines them into a single image-authenticity
pipeline.

``` text
                         Input Face Image
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
          MobileNetV2      EfficientNetB0    LBP + SVM
          2D Forgery         Deepfake       3D Mask Spoof
          Detection          Detection        Detection
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ▼
                    Specialist Decision Engine
                                │
                                ▼
                        Real / Fake Image
```

Each specialist produces spoof evidence for its respective detection
task. The inference layer normalizes the model outputs into a common
spoof-confidence representation before making the final decision.

------------------------------------------------------------------------

# 3D Mask Spoofing Detection

The primary component developed in this project focuses on detecting
**3D mask presentation attacks** through facial micro-texture analysis.

A physical mask attempts to reproduce the appearance of a genuine face,
but differences in surface texture can remain at a local level. This
module therefore focuses on **micro-texture rather than only global
facial appearance**.

The implemented pipeline is:

``` text
              Input Face Image
                     │
                     ▼
             Grayscale Conversion
                     │
                     ▼
          Total Variation Denoising
                     │
                     ▼
                128 × 128
                     │
                     ▼
                  CLAHE
                     │
                     ▼
          Uniform Local Binary Pattern
             Radius = 3
             Points = 24
                     │
                     ▼
          Normalized LBP Histogram
                     │
                     ▼
                RBF SVM
                     │
                     ▼
              Real / Mask Spoof
```

------------------------------------------------------------------------

## Dataset

The 3D mask detection module uses two sources of data:

-   **Labeled Faces in the Wild (LFW)** for genuine face images
-   Frames extracted from videos of individuals wearing **silicone
    masks** for spoof samples

Spoof frames are extracted from the source videos and augmented to
balance the classes and reduce classifier bias.

The preprocessing and data-generation pipeline includes:

-   Real-face image extraction
-   Spoof-video frame extraction
-   Spoof-image augmentation
-   Image preprocessing
-   LBP feature extraction
-   Feature shuffling
-   SVM training and evaluation

------------------------------------------------------------------------

## Why Local Texture?

A mask may reproduce the overall geometry and appearance of a face while
still exhibiting different local surface characteristics.

**Local Binary Patterns (LBP)** provide a compact representation of
local texture by comparing a pixel with its surrounding neighbourhood
and encoding the resulting local intensity pattern.

This makes LBP suitable for representing the subtle texture differences
that can occur between genuine facial skin and artificial mask surfaces.

------------------------------------------------------------------------

## Image Preprocessing

Before feature extraction, the images pass through a preprocessing
pipeline designed to produce a consistent representation for the texture
classifier.

### 1. Grayscale Conversion

The input image is converted to grayscale because the LBP descriptor
operates on local intensity patterns.

### 2. Total Variation Denoising

Total Variation denoising is applied to suppress noise while preserving
important image structures and edges.

### 3. Spatial Normalization

Images are resized to:

``` text
128 × 128
```

### 4. CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to
enhance local contrast before extracting texture features.

### 5. LBP Feature Extraction

Uniform LBP is computed using:

``` text
Radius          = 3
Sampling points = 24
Method          = uniform
```

The resulting LBP codes are converted into a normalized histogram.

For uniform LBP with 24 sampling points, the resulting histogram
contains:

``` text
24 + 2 = 26 bins
```

The histogram forms the feature vector supplied to the SVM classifier.

------------------------------------------------------------------------

# RBF-SVM Classifier

The extracted LBP histograms are classified using a Support Vector
Machine with a **Radial Basis Function (RBF) kernel**.

``` python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

The classifier uses:

``` text
0 → Real
1 → Spoof
```

The dataset is divided using an **80/20 stratified train-test split**
with a fixed random state for reproducibility.

For inference, the SVM's class-1 probability is used as the spoof score:

``` python
predict_proba(features)[0][1]
```

Since class `1` represents spoof, this probability directly corresponds
to the model's spoof-class score.

------------------------------------------------------------------------

# 3D Mask Detection Results

The reported test-set performance of the mask spoofing module is:

| Metric | Score |
|---|---:|
| Accuracy | **95.63%** |
| Precision | **96.00%** |
| Recall | **96.00%** |
| F1 Score | **96.00%** |

------------------------------------------------------------------------

# Other Detection Modules

## 2D Image Forgery Detection --- MobileNetV2

The system includes a MobileNetV2-based specialist for detecting 2D
image forgery.

The model incorporates:

-   Global average pooling
-   Batch normalization
-   Dropout
-   Sigmoid binary classification
-   Fine-tuning of deeper layers
-   Data augmentation

Images are resized to:

``` text
224 × 224
```

and normalized to the `[0, 1]` range.

### Reported Performance

| Metric | Score |
|---|---:|
| Accuracy | **97.90%** |
| Precision | **99.69%** |
| Recall | **95.86%** |
| F1 Score | **97.73%** |

------------------------------------------------------------------------

## Deepfake Detection --- EfficientNetB0

The second deep-learning specialist uses **EfficientNetB0** for
detecting synthetic or manipulated faces.

The module uses the **140K Real and Fake Faces Dataset**, containing
real faces and synthetic faces generated using StyleGAN.

The model uses a fine-tuned EfficientNetB0 backbone with additional
dense layers, batch normalization, LeakyReLU activations, and dropout.

### Reported Performance

| Metric | Score |
|---|---:|
| Accuracy | **98.23%** |
| Precision | **98.62%** |
| Recall | **97.83%** |
| F1 Score | **98.22%** |

------------------------------------------------------------------------

# Specialist-Based Decision Fusion

The original project combined the three model outputs through a simple
average followed by a fixed threshold.

The current inference layer instead treats the models as
**specialists**.

Each model output is first converted into a common:

``` text
spoof_confidence
```

representation, where a larger value consistently indicates stronger
evidence of spoofing.

The output directions are normalized as follows:

  Model            Raw Output Meaning   Spoof Confidence
  ---------------- -------------------- ------------------
  MobileNetV2      High → Spoof         Raw score
  EfficientNetB0   High → Real          `1 - raw score`
  RBF-SVM          High → Spoof         `P(class=1)`

This prevents models with opposite output directions from being
incorrectly interpreted as equivalent scores.

------------------------------------------------------------------------

## Decision Policy

The inference layer uses configurable thresholds:

``` python
SPOOF_THRESHOLDS = {
    "2d_forgery": 0.65,
    "deepfake": 0.65,
    "mask": 0.65
}

BORDERLINE_THRESHOLD = 0.35
```

Each specialist is assigned one of three states:

``` text
Spoof score < 0.35
        ↓
    NO_SPOOF

0.35 ≤ score < 0.65
        ↓
    BORDERLINE

Spoof score ≥ 0.65
        ↓
   STRONG_SPOOF
```

### Final Decision

The current inference policy follows a specialist-triggered rule:

``` text
ANY specialist produces STRONG_SPOOF
                │
                ▼
           Fake Image
```

Therefore, a strong spoof signal from one specialist is sufficient to
trigger a spoof verdict even if the other specialists do not detect the
same attack.

If no specialist reaches the strong-spoof threshold but one or more
specialists produce borderline evidence:

``` text
No STRONG_SPOOF
       +
BORDERLINE evidence
       ↓
Real Image
with BORDERLINE diagnostic state
```

If all specialists produce low spoof scores:

``` text
All specialists < 0.35
        ↓
   Real Image
   CONSISTENT
```

The thresholds are application-level decision parameters and have not
been statistically calibrated or experimentally optimized.

------------------------------------------------------------------------



# Web Application

The project includes a Flask-based web application for image submission
and prediction.

### Application Flow

``` text
Upload Image
     │
     ▼
Flask /predict endpoint
     │
     ├──────────────┬───────────────┐
     ▼              ▼               ▼
 MobileNet      EfficientNet       SVM
     │              │               │
     └──────────────┴───────────────┘
                    │
                    ▼
          Output Normalization
                    │
                    ▼
        Specialist Decision Engine
                    │
                    ▼
             Final Prediction
```

The prediction endpoint accepts an image using:

``` text
POST /predict
```

with the multipart form field:

``` text
image
```

The response includes the individual model scores as well as
specialist-level diagnostic information.

------------------------------------------------------------------------

# Repository Structure

``` text
Image-Spoof-Detection/
│
├── app.py
│
├── mobilenet_model.h5
├── efficientnet_model.h5
├── svm_model.joblib
│
├── mobileNet.ipynb
├── efficientNet.ipynb
├── svm.py
│
├── Course Project Report.pdf
│
├── homepage.png
├── results.png
│
├── README.md
└── requirements.txt
```

### Key Files

  ---------------------------------------------------------------------
  File                               Description
  ---------------------------------- ----------------------------------
  `app.py`                           Flask application and
                                     inference/decision layer

  `svm.py`                           LBP feature-extraction and RBF-SVM
                                     training/evaluation pipeline

  `mobileNet.ipynb`                  MobileNetV2 training notebook

  `efficientNet.ipynb`               EfficientNetB0 training notebook

  `svm_model.joblib`                 Trained RBF-SVM model

  `mobilenet_model.h5`               Trained MobileNetV2 model

  `efficientnet_model.h5`            Trained EfficientNetB0 model

  `Course Project Report.pdf`        Course project documentation
  
  ---------------------------------------------------------------------

------------------------------------------------------------------------

# Running the Application

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

Start the Flask application:

``` bash
python app.py
```

The application runs locally at:

``` text
http://127.0.0.1:5000
```

Upload a supported image through the web interface to obtain the
prediction and detector-level results.

------------------------------------------------------------------------

# Example Interface

## Home Page

![Home Page](homepage.png)

## Results Page

![Results Page](results.png)

------------------------------------------------------------------------

# Reported Module Results

The independently evaluated detection modules reported the following test-set results:

| Module | Model | Accuracy | Precision | Recall | F1 |
|:---|:---|---:|---:|---:|---:|
| 2D Forgery Detection | MobileNetV2 | 97.90% | 99.69% | 95.86% | 97.73% |
| Deepfake Detection | EfficientNetB0 | 98.23% | 98.62% | 97.83% | 98.22% |
| 3D Mask Detection | LBP + RBF-SVM | **95.63%** | **96.00%** | **96.00%** | **96.00%** |

------------------------------------------------------------------------

# Key Takeaways

-   Built a modular face anti-spoofing system combining **deep learning
    and classical machine learning**.
-   Developed a **micro-texture-based 3D mask detection pipeline** using
    LBP features and an RBF-SVM.
-   Applied Total Variation denoising and CLAHE before local texture
    extraction.
-   Extracted uniform LBP descriptors using a radius of 3 and 24
    sampling points.
-   Trained an RBF-SVM with `C=1.0` and `gamma="scale"`.
-   Integrated MobileNetV2, EfficientNetB0, and SVM-based specialists
    into a unified inference pipeline.
-   Normalized heterogeneous model outputs into a common spoof-evidence
    representation.
-   Implemented specialist-triggered decision fusion while retaining
    per-detector diagnostic information.
