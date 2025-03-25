<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/yassine-ben-zekri-72aa6b199/)
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/oumayma-bejaoui-8a6398235/)
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/rami-gharbi)

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <h1 style="font-size:35px">RenalNet</h1>
    <br>
    <p style="font-size:20px" align="center">
        A deep learning solution to detect renal failure from kidney ultrasound images. <br>
        Built for the IEEE ESPRIT SB Code To Cure Hackathon 2025.
    <br>
    <br>
    <a href="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
  <br><br>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME">
    <img src="./assets/logo.png" alt="Logo" width="256px">
  </a>
</div>

# RenalNet: Kidney Failure Detection from Ultrasounds

## Overview
**RenalNet** is an end-to-end machine learning solution designed to predict renal failure from ultrasound images. This project leverages advanced deep learning techniques, custom data augmentation, and robust training strategies to build a clinical diagnostic tool. Developed for the IEEE ESPRIT SB Code To Cure Hackathon 2025, it targets the unique needs of African healthcare by using locally sourced data (from the Cameroon Health System) and innovative AI methods.

The key objectives are:
- To predict the probability of renal failure using ultrasound images.
- To provide a tool that can support clinical decision-making in remote and resource-limited settings.
- To establish a foundation for future mobile applications that monitor kidney health over time.

---

## Detailed Documentation

### 1. Code Structure and Components

#### 1.1. Configuration (CFG)
The `CFG` class centralizes all configuration settings, including:
- **System settings:** Random seed, device (GPU/CPU), number of workers.
- **Data paths:** Locations for images, training CSV, and test CSV.
- **Model hyperparameters:** Image size, batch size, dropout rate, and other training parameters.
- **Training strategy:** Learning rates for head and backbone, number of epochs, and early stopping criteria.
- **Class imbalance handling:** Parameters for mixup, cutmix, and focal loss.

This design ensures that all parameters are easy to manage and update.

#### 1.2. Dataset & Augmentation
- **UltrasoundDataset:**  
  A custom PyTorch Dataset class that:
  - Reads image IDs and labels from a CSV file.
  - Loads ultrasound images from disk, converts them to grayscale (since ultrasounds are typically single-channel), and applies augmentations.
  - Uses Albumentations for robust image transformations (e.g., resizing, flipping, elastic transformations, brightness/contrast adjustments, and coarse dropout).

- **get_augmentations:**  
  A function that returns different augmentation pipelines for training and testing.  
  - **Training pipeline:** Includes aggressive augmentations to improve model generalization.
  - **Testing pipeline:** Focuses on resizing and normalization.

#### 1.3. Model Architecture
- **MedicalImageClassifier:**  
  The core model uses a pre-trained model from the `timm` library (specifically, `'tf_mobilenetv3_small_075'`) modified for:
  - **Grayscale input:** Configured to accept single-channel images.
  - **Attention mechanism:** A convolutional layer with a sigmoid activation that weighs features from the backbone.
  - **Customized head:** A fully connected head with batch normalization, dropout, and spectral normalization to improve training stability.  
  The network outputs a probability (between 0 and 1) indicating the likelihood of renal failure.

#### 1.4. Training Engine & Data Augmentation Techniques
- **Mixup & Cutmix:**  
  Two data augmentation strategies are implemented to reduce overfitting by blending training samples:
  - **Mixup:** Linearly combines pairs of images and their labels.
  - **Cutmix:** Replaces a portion of an image with a patch from another image.
  
- **Focal Loss:**  
  A custom loss function designed to focus on hard examples and address class imbalance. It down-weights well-classified examples so that the model learns more from the difficult ones.

- **Training Loop:**  
  The training loop iterates over epochs, applying mixup/cutmix augmentations randomly. The model is trained using:
  - **AdamW optimizer:** With different learning rates for the backbone (frozen initially) and head.
  - **Learning Rate Scheduler:** Adjusts the learning rate based on the validation AUC.
  - **Early Stopping:** Halts training if no improvement is observed for a specified number of epochs.

#### 1.5. K-Fold Cross-Validation & Ensemble
- **RepeatedStratifiedKFold:**  
  This strategy splits the data into multiple folds ensuring that each fold maintains the same class distribution. Each fold trains a separate model.
  
- **Model Ensemble:**  
  After training, predictions are obtained from each fold using Test Time Augmentation (TTA). The final prediction is the average of predictions from all folds, boosting robustness and performance.

#### 1.6. Inference & Test Time Augmentation (TTA)
- **predict_tta:**  
  This function runs multiple forward passes (TTA steps) on each test image, averages the predictions, and generates a final prediction per image.
  
- **Submission Generation:**  
  The final ensemble predictions are saved to a CSV file in the required format.

---

### 2. Techniques and Innovations

- **Data Augmentation:**  
  Extensive use of Albumentations to generate diverse training samples. This includes spatial transformations (flips, elastic transforms) and pixel-level adjustments (brightness/contrast, blurring).

- **Mixup & Cutmix Augmentation:**  
  By mixing images and labels, the network becomes more robust to noisy labels and prevents overfitting.

- **Attention Mechanism:**  
  Helps the model focus on the most informative parts of the ultrasound images, thereby improving feature extraction and overall performance.

- **Spectral Normalization:**  
  Applied to the final linear layer to stabilize the training process and mitigate exploding gradients.

- **Focal Loss:**  
  A loss function that dynamically scales the standard binary cross-entropy loss, focusing learning on difficult, misclassified examples.

- **K-Fold Cross-Validation and Ensembling:**  
  Using RepeatedStratifiedKFold ensures that the model’s performance is robust across different data splits. The ensemble of multiple folds with TTA further enhances predictive accuracy.

---

### 3. Running the Code

#### 3.1. Data Preparation
- Organize your dataset with the images and CSV files as specified in the configuration.
- Ensure the images are located in the directory defined by `CFG.data_path`, and CSV files (`Train.csv` and `Test.csv`) are in place.

#### 3.2. Training
- Run the main script to start training with cross-validation:
  ```bash
  python RenalNet_Training.ipynb
