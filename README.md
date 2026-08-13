# POLYCARE — AI-Based PCOS Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-Clinical%20Model-green)
![EfficientNet](https://img.shields.io/badge/EfficientNetB0-Image%20Model-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-red)

---

## About the Project

POLYCARE is an AI-based multimodal PCOS (Polycystic Ovary Syndrome)
detection system that combines clinical patient data and ultrasound
images to predict whether a patient has PCOS.

Instead of relying on a single data source, POLYCARE uses a
**Late Fusion approach** — two independent AI models analyze
clinical data and ultrasound images separately, and their
predictions are combined by a meta-classifier to produce a
final, more reliable diagnosis.

The system also provides **Explainable AI** outputs — SHAP
explanations for clinical predictions and Grad-CAM heatmaps
for ultrasound image predictions — so clinicians can understand
why the model made its prediction, not just what it predicted.

---

## Problem Statement

PCOS affects approximately 1 in 10 women of reproductive age.
Early and accurate detection is critical for timely treatment.
Traditional diagnosis relies on clinical tests, ultrasound scans,
and doctor expertise — a process that can be time-consuming and
subjective. POLYCARE aims to support clinicians with an AI-assisted
decision support system that uses both clinical and imaging data
for more accurate and explainable PCOS detection.

---

## System Architecture
PATIENT INPUT
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
Clinical Data Ultrasound Image
(Lab Values) (Ovarian Scan)
│ │
▼ ▼
XGBoost EfficientNetB0
Classifier CNN
│ │
▼ ▼
Clinical Probability Image Probability
│ │
└─────────┬─────────┘
▼
Late Fusion Model
(MLP Meta-Classifier)
│
▼
Final PCOS Prediction
Probability + Confidence
│
▼
Explainable AI Output
SHAP + Grad-CAM
│
▼
PDF Report Generated


---

## Technologies Used

| Component | Technology |
|---|---|
| Backend Framework | Flask (Python) |
| Clinical ML Model | XGBoost |
| Image Deep Learning | EfficientNetB0 (CNN) |
| Fusion Model | MLP Meta-Classifier (Late Fusion) |
| Clinical Explainability | SHAP (SHapley Additive exPlanations) |
| Image Explainability | Grad-CAM Visualization |
| Clinical Preprocessing | Pandas, Scikit-learn |
| Image Preprocessing | TensorFlow, Keras |
| Frontend | HTML, CSS, JavaScript |
| PDF Report Generation | ReportLab / WeasyPrint |
| Model Serialization | Pickle (.pkl), HDF5 (.h5) |

---

## Project Structure

POLYCARE/
├── app.py # Flask backend — main application
├── preprocess.py # Clinical data preprocessing pipeline
├── explainability.py # SHAP and Grad-CAM generation
├── index.html # Landing page
├── predict.html # Patient data input form
├── result.html # Prediction results and explanations
├── about.html # About the system
├── style.css # Application styling
├── script.js # Frontend JavaScript
├── xgb_model.pkl # Trained XGBoost clinical model
├── imagef_model.h5 # Trained EfficientNetB0 image model
├── structured_model.pkl # Trained late fusion meta-classifier
├── scaler.pkl # Feature scaler for clinical data
├── imputer.pkl # Data imputer for missing values
├── feature_info.json # Clinical feature metadata
├── model_metadata.json # Model performance metadata
├── shap_data.json # SHAP background data
└── polycarelogo.png # Application logo


---

## Key Features

**Multimodal Detection**
Combines clinical lab values and ultrasound images for more
accurate prediction than single-modality approaches.

**Explainable AI**
SHAP explanations show which clinical features contributed
most to the prediction. Grad-CAM heatmaps highlight which
regions of the ultrasound image influenced the CNN's decision.

**Late Fusion Architecture**
Both models predict independently first. Their probabilities
are then combined by a meta-classifier — more robust than
early fusion or single model approaches.

**PDF Report Generation**
Complete clinical report generated with prediction result,
probability scores, SHAP explanations, and Grad-CAM
visualization — suitable for clinical records.

**Flask REST API Backend**
Clean API endpoints for prediction, making the system
extensible and integration-ready.

---

## Clinical Parameters Used

The XGBoost clinical model uses the following patient parameters:

- Age and BMI
- Menstrual cycle length and regularity
- LH (Luteinizing Hormone) and FSH (Follicle Stimulating Hormone)
- LH/FSH ratio
- AMH (Anti-Müllerian Hormone)
- Testosterone levels
- Follicle count and size
- Weight and related measurements
- Other hormonal markers

---

## Evaluation Metrics

The system is evaluated on:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC Score
- Confusion Matrix

For a medical classification system, accuracy alone is
insufficient. Sensitivity (recall) is particularly critical
to minimize false negatives — cases where PCOS is present
but not detected.

---

## Testing Approach

**Functional Testing**
- Verified clinical data input form accepts valid lab values
- Verified system rejects invalid or out-of-range lab values
- Verified ultrasound image upload accepts correct formats
- Verified prediction results display correctly
- Verified PDF report generates with all required fields

**Integration Testing**
- Verified Flask API correctly receives clinical data and
  passes it to preprocessing pipeline
- Verified preprocessed data correctly feeds into XGBoost model
- Verified image preprocessing correctly feeds into EfficientNetB0
- Verified both model outputs correctly feed into fusion model
- Verified SHAP and Grad-CAM outputs correctly appear in results

**Edge Case Testing**
- Verified system handles missing or incomplete lab values
- Verified system handles non-ultrasound image uploads gracefully
- Verified system handles boundary lab values correctly
- Verified PDF generation completes within acceptable time

**Validation**
- Clinical model validated on unseen test data
- Image model validated on held-out test set
- Fusion model validated end-to-end with complete patient data

---

## How to Run

**Prerequisites**
```bash
Python 3.8+
pip install flask xgboost tensorflow scikit-learn
pip install shap pandas numpy pillow reportlab
```

**Clone the repository**
```bash
git clone https://github.com/sinchana1403/POLYCARE.git
cd POLYCARE
```

**Run the Flask application**
```bash
python app.py
```

**Open in browser**

http://localhost:5000


---

## Internship and Academic Context

This project was developed as the final year capstone
project for Bachelor of Computer Application (BCA) at
R R Institute of Management Studies, Bengaluru (2026).

It was also used as a practical demonstration of software
testing concepts including functional testing, integration
testing, edge case testing, and API validation during
the author's software testing internship.

---

## Author

**Sinchana**  
BCA Graduate — R R Institute of Management Studies, Bengaluru  
Fresher QA Engineer | Python | Flask | Machine Learning  
GitHub: https://github.com/sinchana1403

---

## Disclaimer

POLYCARE is an AI-assisted decision support system developed
for academic purposes. It is not intended to replace
professional medical diagnosis. All predictions should be
reviewed and validated by qualified medical professionals.

