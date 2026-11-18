# Deep Learning for Video Action Recognition on UCF101

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-VideoMAE-FFD21E)
![Dataset](https://img.shields.io/badge/Dataset-UCF101-blue)
![University](https://img.shields.io/badge/University%20of%20Verona-M.Sc.%20Artificial%20Intelligence-8B0000)

A systematic comparative analysis of deep learning architectural paradigms for **Human Action Recognition (HAR)** on the **UCF101** video dataset, tracking the technological evolution from static spatial feature aggregation (*Bag-of-Frames*) to sequential recurrent modeling (*CNN + LSTM / BiLSTM*) and self-supervised spatiotemporal Transformers (*VideoMAE*).

> 📄 **Full Technical Report:** See [`report_final.pdf`](./report_final.pdf) for the complete theoretical background, architectural diagrams, training curves, and per-class performance analysis.

---

## 📌 Project Overview & Objectives

While deep learning has largely mastered static image classification, video action understanding remains challenging due to the complex interplay between **spatial visual context** and **temporal motion dynamics**. Focusing exclusively on **RGB input** (avoiding the heavy computational and storage overhead of pre-computing Optical Flow), this study investigates the transition from *"scene recognition"* to *"true action understanding"* across three progressive modeling phases:

1. **Spatial Baselines (*Bag-of-Frames*)**: Establish a performance floor using 2D CNN backbones with temporal mean pooling to measure how much information can be extracted solely from static visual appearance and background context.
2. **Recurrent Modeling (*CNN + LSTM / BiLSTM*)**: Quantify the *"value of time"* by processing frame-level feature sequences through unidirectional and bidirectional recurrent networks to capture chronological motion dynamics.
3. **Attention Mechanisms (*Vision Transformers*)**: Evaluate whether global spatiotemporal self-attention and masked autoencoding (**VideoMAE**) capture long-range dependencies more effectively than recurrent hidden states under single-view inference constraints.

Beyond global Top-1 accuracy, a granular **per-class evaluation** is conducted to distinguish **spatially dominated actions** (identifiable from single frames/environments) from **temporally sensitive actions** (requiring sequential motion reasoning).

---

## 📂 Repository Structure

```text
Action-Recognition-on-UCF101/
├── notebooks/
│   ├── note.ipynb                      # Preprocessing script: Video (.avi) to 224x224 JPEG frame extraction & archiving
│   ├── dataset.ipynb                   # Exploratory Data Analysis (EDA), class distribution, and DataLoader verification
│   ├── training_CustomCNN.ipynb        # Training & evaluation of the scratch Custom 2D CNN baseline
│   ├── fine_tuning_ResNet18.ipynb      # Fine-tuning & evaluation of ResNet-18 with Temporal Mean Pooling
│   ├── training_customCNN_LSTM.ipynb   # Training & evaluation of CustomCNN + LSTM hybrid architecture
│   ├── ResNet2D_LSTM.ipynb             # End-to-end training & evaluation of ResNet-18 + Unidirectional LSTM
│   ├── ResNet2D_BiLSTM.ipynb           # End-to-end training & evaluation of ResNet-18 + Bidirectional LSTM
│   └── ViT_training.ipynb              # Fine-tuning & evaluation of the pretrained VideoMAE-base Transformer
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── dataset2.py                     # UCF101DatasetFast loader, group-wise stratified split, and data integrity check
│   ├── models.py                       # PyTorch implementations of CustomCNN, ResNet2D_Baseline, CustomCNN_LSTM, ResNet2D_LSTM, and ResNet2D_BiLSTM_Pro
│   └── training.py                     # Training/validation epoch loops and automatic experiment run management
├── .gitignore                          # Git ignore rules for datasets, caches, and checkpoints
├── README.md                           # Project documentation
├── report_final.pdf                    # Final academic report (M.Sc. in Artificial Intelligence, University of Verona)
└── requirements.txt                    # Python dependencies
```

---

## 🛠️ Methodology & Data Pipeline

### 1. The UCF101 Dataset
The [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) consists of **13,320 video clips** across **101 human action categories** grouped into five macro-categories: *Human-Object Interaction*, *Body-Motion Only*, *Human-Human Interaction*, *Playing Musical Instruments*, and *Sports*. Within each class, clips are organized into **25 video groups** (4–7 clips per group) sharing common subjects, backgrounds, and camera viewpoints.

### 2. Group-Wise Splitting & Integrity Verification
To prevent **data leakage** (where a model achieves artificially high validation scores by memorizing background scenes or actors shared across clips of the same group), the validation set (20%) is extracted from the official training split (`trainlist01.txt`) using a **stratified group-wise splitting strategy** (`Archery_g01`, `Archery_g02`, etc.) implemented in `src/dataset2.py`:
- **Training Set:** 7,675 video clips
- **Validation Set:** 1,862 video clips
- **Test Set (`testlist01.txt`):** 3,783 video clips
- Automated set-intersection checks verify **zero overlapping clips or groups** across splits.

### 3. I/O Optimization, Temporal Sampling & Augmentation
- **JPEG Frame Extraction (`notebooks/note.ipynb`):** Native `.avi` clips are decoded offline and stored as $224 \times 224$ JPEG frame directories, eliminating video decoding bottlenecks during GPU training.
- **Temporal Depth ($T = 16$ frames):** Clips are sampled to a fixed sequence length of $T = 16$ frames, producing input tensors of shape `(Batch, 3, 16, 224, 224)` (permuted to `(Batch, 16, 3, 224, 224)` for VideoMAE).
- **Spatial Transformations:**
  - *Training:* Resize to $224 \times 224$, Random Horizontal Flip ($p = 0.5$), Color Jitter (brightness = 0.1, contrast = 0.1), and ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - *Validation / Test:* Resize to $224 \times 224$ and ImageNet normalization under a **single-view, single-crop** evaluation protocol.

---

## 🧠 Implemented Architectures

| Model | Paradigm | Spatial Backbone | Temporal Aggregation / Modeling | Key Regularization & Optimization |
| :--- | :--- | :--- | :--- | :--- |
| **`CustomCNN`** | Spatial Baseline (*Bag-of-Frames*) | 4-block 2D CNN ($32 \to 64 \to 128 \to 256$) + BatchNorm + MaxPool + GAP (from scratch) | Temporal Mean Pooling across $T=16$ frames | Dropout (0.5), Adam ($\eta = 10^{-3}$), Early Stopping |
| **`ResNet2D_Baseline`** | Transfer Learning (*Bag-of-Frames*) | ImageNet-pretrained **ResNet-18** (`conv1` through `layer2` frozen; `layer3`–`layer4` fine-tuned) | Temporal Mean Pooling across $T=16$ frames | Dropout (0.5), Adam ($\eta = 10^{-5}$), Early Stopping |
| **`CustomCNN_LSTM`** | Scratch Hybrid Recurrent | Pretrained `CustomCNN` feature extractor ($256$-d embeddings) | 1-layer Unidirectional **LSTM** ($H = 256$), last time-step classification | Dropout (0.5), Adam |
| **`ResNet2D_LSTM`** | Sequential Recurrent | ImageNet-pretrained **ResNet-18** ($512$-d frame features, full fine-tuning) | 2-layer Unidirectional **LSTM** ($H = 1024$) + Temporal Mean Pooling | Spatial Dropout (0.5), Label Smoothing (0.1), Differential LR ($\eta_{\text{backbone}} = 10^{-5}$, $\eta_{\text{head}} = 10^{-4}$), AdamW, `ReduceLROnPlateau` |
| **`ResNet2D_BiLSTM_Pro`** | Bidirectional Recurrent | ImageNet-pretrained **ResNet-18** ($512$-d frame features, full fine-tuning) | 2-layer Bidirectional **BiLSTM** ($H = 512 \times 2 = 1024$) + Temporal Mean Pooling | Spatial Dropout (0.6), LSTM/MLP Dropout (0.5), Label Smoothing (0.1), Gradient Clipping (1.0), Differential LR, AdamW |
| **`VideoMAE` (Base)** | Spatiotemporal Transformer | Self-supervised **ViT-Base** (`MCG-NJU/videomae-base`, pretrained on Kinetics-400 for 1600 epochs) | 3D Tubelet Embeddings + Joint Spatiotemporal Self-Attention | Label Smoothing (0.1), AdamW ($\eta = 2 \times 10^{-5} \sim 5 \times 10^{-5}$), Early Stopping |

---

## 📊 Experimental Results & Comparative Analysis

### Overall Performance Summary (UCF101 Split-1, RGB-Only, Single-View)

| Architecture | Paradigm | Pretraining | Validation Accuracy | Test Accuracy | Notes / Comparison |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Custom 2D CNN** | Spatial (*Bag-of-Frames*) | None (From Scratch) | ~29.6% | **27.76%** | Severe early overfitting; insufficient representational capacity |
| **ResNet-18 + Mean Pooling** | Spatial (*Bag-of-Frames*) | ImageNet-1K | ~75.8% | **75.71%** | Strong scene/context classifier, but blind to chronological motion |
| **ResNet-18 + LSTM** | Unidirectional Recurrent | ImageNet-1K | ~75.2% | **73.88%** | Unlocks purely dynamic actions (*HandstandWalking*: $0\% \to 20.59\%$) |
| **ResNet-18 + BiLSTM** | Bidirectional Recurrent | ImageNet-1K | ~78.0% | **78.14%** | **+7.04%** over published RGB-only LRCN benchmark (71.1%) |
| **VideoMAE (Base)** | Spatiotemporal Transformer | Kinetics-400 (SSL) | **82.92%** | **79.00%** | Highest capacity & validation peak under single-view evaluation |

### Key Findings & Per-Class Insights
1. **Static Scene Recognition vs. Temporal Dynamics:**
   - The **ResNet-18 Bag-of-Frames** baseline achieves **100% accuracy** on context-rich classes such as *SumoWrestling*, *Skijet*, *Surfing*, *Rafting*, and *VolleyballSpiking*, where static background cues suffice. However, due to *"temporal collapse"*, it completely fails on motion-defined classes like *HandstandWalking* (**0.00%**) and *JumpRope* (**2.63%**).
2. **The Value of Bidirectional Recurrence:**
   - Introducing the **2-layer BiLSTM** alongside stratified differential learning rates ($\eta = 10^{-5}$ for the ResNet backbone, $\eta = 10^{-4}$ for recurrent/classifier layers) and strong spatial dropout ($0.6$) elevates test accuracy to **78.14%**. Difficult motion-driven categories improve substantially (e.g., *JumpRope* rises from **2.63%** to **26.32%**, *Nunchucks* from **11.43%** to **37.14%**, and *BodyWeightSquats* from **10.00%** to **40.00%**).
3. **Transformer Capacity & Single-View Evaluation Gap:**
   - **VideoMAE-base** achieves the best overall performance (**82.92%** validation accuracy, **79.00%** test accuracy). The gap compared to the multi-view literature benchmark (96.1%) is explained by two deliberate experimental choices:
     - **Single-View Inference:** Prioritizing computational efficiency by evaluating a single 16-frame clip per video rather than 12-view ensemble averaging (3 temporal clips $\times$ 4 spatial crops).
     - **Regularization Bottleneck:** High-capacity Vision Transformers quickly saturate on medium-scale datasets without heavy mix-based augmentations (*Mixup*, *CutMix*, *RandAugment*).

---

## 👤 Author

* **Riccardo Fidanza** (ID: `VR516130`)
* **Degree:** Master's Degree in Artificial Intelligence
* **Course:** Deep Learning (Academic Year 2025–2026)
* **Institution:** [University of Verona](https://www.univr.it/)

---

## 📚 References

1. **Soomro, K., Zamir, A. R., & Shah, M.** (2012). *UCF101: A dataset of 101 human actions classes from videos in the wild*. arXiv preprint arXiv:1212.0402.
2. **Donahue, J., et al.** (2015). *Long-term recurrent convolutional networks for visual recognition and description*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2625–2634.
3. **Wang, L., et al.** (2016). *Temporal segment networks: Towards good practices for deep action recognition*. In European Conference on Computer Vision (ECCV), pp. 20–36.
4. **Tong, Z., Song, Y., Wang, J., & Wang, L.** (2022). *VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training*. In Advances in Neural Information Processing Systems (NeurIPS).
