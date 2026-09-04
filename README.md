# Integrated Deep Learning Framework for Fishermen Assistance through Fish Detection and Fishing Zone Prediction

##  Project Status

**Ongoing**

The project currently focuses on deep-learning-based fish detection, fish abundance analysis, large-fish/species identification, live detection, and Explainable AI (XAI) using LayerCAM.

The **Fishing Zone Prediction** module is currently under development and will be integrated into the system in a future stage.

---

##  Overview

This project aims to develop an integrated deep learning framework to assist fishermen by providing information about fish presence and abundance from underwater or marine images.

The system uses YOLO11-based object detection to identify fish and estimate their abundance. A separate detection model is used to identify large fish species such as Tuna and Shark.

To improve model interpretability, **LayerCAM** is incorporated to generate visual explanations showing the regions of the image that contribute to the fish detection.

The project also includes a web-based interface developed using Flask for image-based detection, live detection, and visualization of model results.

The final system is planned to include a Fishing Zone Prediction module using environmental and satellite-derived parameters.

---

##  Current Features

### 1.  Fish Detection

A YOLO11-based object detection model is used to detect fish in input images.

The system provides:

- Fish presence detection
- Fish counting
- Bounding-box visualization
- Confidence scores
- Fish abundance estimation

### 2.  Fish Abundance / Density

Based on the detected fish count, the system categorizes fish abundance into:

- **Low**
- **Medium**
- **High**

### 3.  Large Fish / Species Detection

A separate YOLO11 model is used for identifying large fish species.

Currently supported species include:

- Tuna
- Shark

### 4.  Explainable AI using LayerCAM

**LayerCAM** is used to provide visual explanations of the fish detection model.

The generated heatmap highlights the regions that contribute to the model's prediction, helping users understand where the model is focusing when detecting fish.

The explanation is generated for individual detected fish using an object-centric approach.

### 5.  Live Detection

The system supports live fish detection through a video stream.

A phone screen can be mirrored using **scrcpy**, captured by the application, and processed for real-time fish detection.

### 6.  Web Interface

A Flask-based web application provides separate interfaces for:

- Home
- Fish Detection
- Live Detection
- LayerCAM/XAI
- Detection History
- About

---

##  Deep Learning Models

### Fish Detection Model

- Architecture: **YOLO11**
- Task: Fish object detection
- Classes: Single fish class
- Input images: Marine/underwater fish images

### Large Fish Detection Model

- Architecture: **YOLO11**
- Task: Large fish/species detection
- Classes:
  - Tuna
  - Shark

---

##  Model Performance

### Fish Detection Model

| Metric | Result |
|---|---:|
| mAP50 | 0.971 |
| mAP50-95 | 0.760 |
| Precision | 0.925 |
| Recall | 0.945 |
| Presence Accuracy | 98.6% |

---

##  LayerCAM Visualization

LayerCAM is applied to the fish detection model to generate visual explanations.


The pipeline is:
```text
Input Image
     ↓
YOLO11 Fish Detection
     ↓
Fish Bounding Boxes
     ↓
Object-Centric Crop
     ↓
LayerCAM
     ↓
Fish Silhouette Refinement
     ↓
Heatmap Generation
     ↓
Final Explanation
```

##  Web Application Workflow

The web application workflow is:

```text
Marine Image
     ↓
Flask Web Interface
     ↓
Image Preprocessing
     ↓
YOLO11 Fish Detection
     ↓
Fish Count & Analysis
     ↓
Abundance / Density Estimation
     ↓
Large Fish / Species Detection
     ↓
Result Display
```

### XAI Workflow

```text
Input Image
     ↓
YOLO11 Fish Detection
     ↓
Detected Fish
     ↓
Object-Centric Crop
     ↓
LayerCAM
     ↓
Fish Silhouette Refinement
     ↓
Heatmap Generation
     ↓
XAI Visualization
```

##  Technologies Used

- Python
- YOLO11
- PyTorch
- Ultralytics
- OpenCV
- LayerCAM
- Flask
- Flask-CORS
- HTML
- CSS
- JavaScript
- NumPy
- Pillow
- scrcpy



