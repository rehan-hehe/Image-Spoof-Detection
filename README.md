# Image Spoof Detection System

This repository contains the source code for a multi-stage Image Spoof Detection System, developed as a course project for DA-221M under Dr Teena Sharma. The system integrates deep learning models (MobileNet, EfficientNet) with traditional machine learning (SVM on LBP features) to create a robust pipeline for detecting various types of image manipulation. The entire system is served via a user-friendly Flask web application.

**For a detailed breakdown of the methodology, datasets, and results, please see the full `Course Project Report.pdf` included in this repository.**
----

## Core Features
-   **Multi-Stage Detection Pipeline:** Defends against various attacks including 2D image forgery, deepfakes, and 3D mask spoofing.
-   **Hybrid ML/DL Approach:** Combines the feature-rich analysis of CNNs with the micro-texture analysis of SVMs on LBP features.
-   **Interactive Web Interface:** Allows users to upload an image and receive a real-time authenticity score from the ensemble model.

## System Architecture & Performance
The system is built from three distinct detection modules. Each was trained and evaluated independently before being integrated.

| Detection Module       | Key Models Used        | F1-Score |
| ---------------------- | ---------------------- | :------: |
| **2D Image Forgery**   | MobileNetV2            | 97.73%   |
| **Deepfake Detection** | EfficientNetB0, Dlib   | 98.22%   |
| **3D Mask Spoofing**   | SVM on LBP features    | 96.00%   |

*This was a team project. My primary contribution was the development of the **3D Mask Spoofing** module.*

----

## Tech Stack
-   **Machine Learning:** TensorFlow, Keras, Scikit-learn
-   **Computer Vision:** OpenCV, Dlib, Scikit-image
-   **Core Libraries:** Python, NumPy

----

## Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/[your-username]/image-spoof-detection-system.git
    cd image-spoof-detection-system
    ```

2.  **Install Dependencies:**
    *(It is recommended to use a virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask Application:**
    ```bash
    # Ensure your trained model files (.h5, .pkl) are in the correct directory
    flask run
    ```
----

## Team Contributions
-   **Module 1 (Web App & Preprocessing):** Bipan Chandra
-   **Module 2 (2D Forgery):** Pratik Ranjan
-   **Module 3 (Deepfake):** Rithvik Ponnapalli
-   **Module 4 (3D Mask Spoofing):** Rehan Sherawat
