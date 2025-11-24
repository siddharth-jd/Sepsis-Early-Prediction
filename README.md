# Early Sepsis Detection Using Deep Learning on ICU Time-Series Data

This project presents a complete deep learning pipeline for **early detection of sepsis** using **ICU time-series vitals and laboratory measurements**.  
The goal is to predict whether a patient will develop sepsis in the **next 6 hours**, enabling clinicians to intervene earlier.

The project includes classical ML baselines (Logistic Regression, Random Forest) and advanced deep learning models (LSTM and Attention-based LSTM) with interpretability.

---

## 🚑 1. Background & Motivation

Sepsis is a life-threatening organ dysfunction caused by a dysregulated response to infection.  
**Early detection is critical** — every hour of delay significantly increases mortality.

Traditional scoring systems such as SOFA and qSOFA rely on fixed thresholds and often fail to generalize.  
Deep learning models, especially on **continuous ICU time-series data**, can recognize subtle patterns long before clinical diagnosis.

---

## 📊 2. Dataset

**Dataset:** Kaggle / PhysioNet-style Sepsis dataset  
**Granularity:** One row = 1 hour of ICU time for a patient  
**Includes:**  
- Physiological vitals (HR, Temp, MAP, SBP, DBP, Resp, SpO₂, etc.)  
- Laboratory values (Lactate, WBC, Platelets, Creatinine, etc.)  
- Demographics (Age, Gender)  
- `SepsisLabel` – 1 from the hour of sepsis onset onwards  
- `Patient_ID` – unique ICU stay identifier  
- `Hour` – time index

### ✔ Derived Target: Early Warning Label  
A new target `EarlyLabel` is generated:  
> **1 if patient will become septic within the next 6 hours, else 0**

This label enables **clinically meaningful early prediction**.

---

## 🛠️ 3. Project Pipeline (Phases)

### **Phase 1 – Data Loading & Understanding**
- Load dataset from Google Drive
- Inspect features, missingness, label distribution

### **Phase 2 – Preprocessing**
- Forward-fill missing values per patient  
- Mean-fill remaining missing values  
- Normalize features using train-only statistics  
- Split by patient: 70% Train / 15% Val / 15% Test  
- Build sequences `(T, F)` per patient with padding masks  
- Create `EarlyLabel` (6-hour prediction window)

### **Phase 3 – Baseline Models**
- Logistic Regression  
- Random Forest  
- Both operate on per-hour flattened features  
- Evaluated with AUROC and AUPRC

### **Phase 4 – Deep Learning Model (LSTM)**
- Timestep-level early sepsis prediction  
- BCE loss with padding masks  
- Per-timestep AUROC/AUPRC evaluation

### **Phase 5 – Full Evaluation**
- ROC curves (train/val/test)  
- Precision-Recall curves  
- Precision at target recall (e.g., 80%)  
- Confusion matrix at optimal threshold

### **Phase 6 – Interpretability (Attention-Based LSTM)**
- Temporal attention over patient’s ICU hours  
- Identify critical timesteps before sepsis onset  
- Visualize attention vs. hour vs. early labels  
- Insights into model reasoning

---

## 🤖 4. Models Implemented

### **Baseline Models**
- **Logistic Regression**
- **Random Forest**

### **Deep Learning Models**
- **LSTM (timestep-level prediction)**
- **Attention-Enhanced LSTM (sequence-level prediction + interpretability)**

All deep models are implemented in **PyTorch** with custom dataloaders and padding logic.

---

## 📈 5. Evaluation Metrics

Because sepsis is rare, standard accuracy is misleading.  
The following metrics are used:

- **AUROC** – discriminative ability  
- **AUPRC** – crucial for imbalanced datasets  
- **Recall @ fixed precision**  
- **Precision @ fixed recall**  
- **Confusion Matrix**  
- **Attention Visualization** (for interpretability)

---

## 📁 6. Repository Structure

```text
.
├── Early_Sepsis_Detection_Project.ipynb    # Main notebook with full pipeline
├── README.md
└── sepsis_project/
    └── Dataset.csv                         # Input dataset
