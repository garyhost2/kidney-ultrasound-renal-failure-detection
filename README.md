<div align="center">

  <h1>🩺 RenalNet</h1>
  <h3>🏆 1st Place – IEEE ESPRIT SB Code To Cure Hackathon 2025</h3>
  <p>A deep learning solution to detect renal failure from kidney ultrasound images.</p>

  <p>
    👤 <a href="https://www.linkedin.com/in/yassine-ben-zekri-72aa6b199/">Yassine Ben Zekri</a> •
    👤 <a href="https://www.linkedin.com/in/oumayma-bejaoui-8a6398235/">Oumayma Bejaoui</a> •
    👤 <a href="https://www.linkedin.com/in/rami-gharbi">Rami Gharbi</a>
  </p>

  <img src="./assets/logo.png" alt="RenalNet Logo" width="180" />

</div>

<hr/>

<h2>🌐 Overview</h2>

<p><strong>RenalNet</strong> is a clinical AI pipeline that classifies ultrasound kidney images to assess renal failure probability.
It combines modern CNN architectures, attention mechanisms, strong augmentations, and ensemble learning to improve robustness in real-world healthcare settings.</p>

<h3>🎯 Goals:</h3>
<ul>
  <li>Predict renal failure using ultrasound imagery.</li>
  <li>Assist diagnosis in low-resource environments.</li>
  <li>Enable future mobile kidney health monitoring tools.</li>
</ul>

<hr/>

<h2>🔁 Pipeline Overview</h2>

<pre>
graph TD
  A[Input: CSV + Ultrasound Images] --> B[UltrasoundDataset (Grayscale)]
  B --> C[Albumentations Augmentation]
  C --> D[MobileNetV3 Backbone + Custom Attention Head]
  D --> E[Training with Mixup, Cutmix, Focal Loss]
  E --> F[K-Fold Cross-Validation + EarlyStopping]
  F --> G[Test Time Augmentation (TTA)]
  G --> H[Final Ensemble Prediction]
  H --> I[Submission CSV Generation]
</pre>

<hr/>

<h2>⚙️ Architecture & Components</h2>

<h3>🧩 1. Configuration</h3>
<ul>
  <li><code>CFG</code> class defines seed, device, paths, batch size, augmentations, and model hyperparameters.</li>
</ul>

<h3>🧩 2. Dataset & Augmentation</h3>
<ul>
  <li>Custom PyTorch dataset to load grayscale ultrasound images.</li>
  <li>Albumentations used for:
    <ul>
      <li>✳️ Training: Resize, flip, elastic distortions, contrast, dropout.</li>
      <li>🔍 Testing: Resize + normalization.</li>
    </ul>
  </li>
</ul>

<h3>🧩 3. Model</h3>
<ul>
  <li>Backbone: <code>tf_mobilenetv3_small_075</code> (via <code>timm</code>)</li>
  <li>Enhanced with:
    <ul>
      <li>✅ Attention mechanism</li>
      <li>✅ Spectral Normalization</li>
      <li>✅ Dropout for regularization</li>
    </ul>
  </li>
</ul>

<h3>🧩 4. Training Strategy</h3>
<ul>
  <li>Augmentations: Mixup + Cutmix applied randomly per epoch.</li>
  <li>Loss: FocalLoss</li>
  <li>Optimizer: AdamW</li>
  <li>LR Scheduler: Reduce on Plateau (AUC-based)</li>
  <li>EarlyStopping enabled</li>
</ul>

<h3>🧩 5. Evaluation</h3>
<ul>
  <li><code>RepeatedStratifiedKFold</code> ensures class balance</li>
  <li>TTA for robust inference</li>
  <li>Ensemble average across folds + augmentations</li>
</ul>

<hr/>

<h2>🚀 Quickstart</h2>

<pre>
# Step 1 — Data Preparation
Place ultrasound images and CSV files:
Train.csv | Test.csv | images/

# Step 2 — Train the Model
python RenalNet_Training.ipynb

# Step 3 — Inference
python Inference_with_TTA.py
</pre>

<hr/>

<h2>💡 Key Innovations</h2>
<ul>
  <li>🔍 Attention mechanism to focus on salient regions</li>
  <li>🧼 SpectralNorm stabilizes training</li>
  <li>🔁 Strong data augmentation with Cutmix + Mixup</li>
  <li>⚖️ FocalLoss handles class imbalance</li>
  <li>🔬 K-Fold with TTA improves generalization</li>
</ul>

<hr/>

<h2>📸 Pipeline Image</h2>

<p><img src="https://i.postimg.cc/02pTMvPd/Chat-GPT-Image-Apr-16-2025-08-05-32-PM.png" width="600" /></p>

<hr/>

<div align="center">
  <strong>Built with ❤️ by the RenalNet Team at IEEE ESPRIT SB</strong>
</div>
