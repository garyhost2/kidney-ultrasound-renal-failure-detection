# 🧪 RenalNet
[![LinkedIn\][linkedin-shield\]](https://www.linkedin.com/in/yassine-ben-zekri-72aa6b199/) [![LinkedIn\][linkedin-shield\]](https://www.linkedin.com/in/oumayma-bejaoui-8a6398235/) [![LinkedIn\][linkedin-shield\]](https://www.linkedin.com/in/rami-gharbi)

---

---

## 🔍 Overview

**RenalNet** is an end-to-end machine learning solution designed to predict renal failure from ultrasound images. It uses deep learning techniques, attention mechanisms, and data augmentation strategies to create a clinically-relevant prediction system, tailored to African healthcare data (e.g., Cameroon Health System).

**Key Goals:**

- Predict renal failure using ultrasound images.
- Assist clinicians in remote or resource-limited regions.
- Lay the foundation for a future mobile kidney health monitoring tool.

---

## 🔄 Pipeline Summary

```mermaid
graph TD
  A[Input CSV + Ultrasound Images] --> B[UltrasoundDataset Class (Grayscale)]
  B --> C[Albumentations Augmentation]
  C --> D[Model: tf_mobilenetv3_small_075]
  D --> E[Attention Layer + SpectralNorm Head]
  E --> F[Focal Loss + AdamW Optimizer]
  F --> G[Repeated Stratified K-Fold Training]
  G --> H[Test-Time Augmentation (TTA)]
  H --> I[Ensemble Averaging]
  I --> J[Submission CSV]
```

---

## 🌐 Components

### 1. Configuration (`CFG` class)

- Controls seed, device (CPU/GPU), data paths, and hyperparameters.
- Allows flexible tuning of augmentations, losses, and optimizer settings.

### 2. Dataset & Augmentations

- **UltrasoundDataset**: Loads grayscale images + applies transformations.
- **Albumentations**:
  - *Train*: Resize, flips, elastic distortions, brightness/contrast, CoarseDropout.
  - *Test*: Resize + normalization.

### 3. Model Architecture

- Pretrained **MobileNetV3-Small (timm)**
- **Grayscale adaptation** + **sigmoid-based attention layer**
- Final linear layer with spectral normalization & dropout.

### 4. Data Augmentation Techniques

- **Mixup** & **Cutmix** (applied randomly during training)
- **Focal Loss** to emphasize hard-to-classify samples

### 5. Training

- **Optimizer**: AdamW with different learning rates for head/backbone
- **Scheduler**: Validation AUC-based LR scheduler
- **EarlyStopping** based on plateaued AUC

### 6. Cross-Validation & Ensemble

- **RepeatedStratifiedKFold** ensures class balance in splits
- **TTA** averages predictions across multiple augmented versions
- **Final Prediction** = mean across folds + TTA outputs

---

## 🔍 Innovations

- **Attention Mechanism** for feature focus
- **SpectralNorm** for stabilized training
- **Heavy Augmentation** improves generalization
- **Focal Loss** addresses class imbalance effectively
- **K-Fold + TTA + Ensemble** = highly robust predictions

---

## 🚀 How to Run

### 1. Prepare Data

- Place ultrasound images & `Train.csv`, `Test.csv` in paths defined in `CFG`

### 2. Train the Model

```bash
python RenalNet_Training.ipynb
```

### 3. Generate Submission

```bash
python Inference_with_TTA.py
```

---

**Hackathon**: IEEE ESPRIT SB - *Code To Cure 2025*\
**Team**: Yassine Ben Zekri, Oumayma Bejaoui, Rami Gharbi

