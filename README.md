# Multi-Model Plant & Tree Identification and Benchmarking Platform
### Programmatic 10-View Test-Time Augmentation (TTA) and Comparative Architecture Analysis
Reference repository: https://github.com/plantnet/PlantNet-300K

**Authors:** Halil Akbaş & Karahan Ballı  
**Academic Presentation & Benchmarking Project**  
**Technology Stack:** PyTorch, Flask, Timm, PIL, Test-Time Augmentation (TTA), HTML5/CSS3/Vanilla JS  
**GitHub Repository:** [DL4-PlantNET300K-ClassifierAnalysis](https://github.com/halilakbas11/DL4-PlantNET300K-ClassifierAnalysis)

---

## 1. Project Vision & Overview

Deep learning has revolutionized botanical taxonomy, enabling instant species recognition from real-world photography. However, there is no "one-size-fits-all" model for deployment: local edge devices and offline mobile apps require lightweight, fast models (e.g., MobileNet), while cloud servers demand maximum accuracy regardless of memory foot-print (e.g., Vision Transformers or deep ResNets).

**This project does not train deep learning models from scratch.** Instead, it integrates high-quality, pre-trained SOTA weights officially released by the Pl@ntNet team on the **Pl@ntNet-300K** benchmark. 

### Core Contributions:
1.  **Multi-Model Comparative Benchmarking Platform**: A dynamic web application that hosts 6 state-of-the-art architectures, letting users hot-swap models in real-time to compare predictions, confidence scores, and latency.
2.  **Memory-Efficient Dynamic Caching**: An on-demand model manager in Flask that loads weights into RAM only when selected via the interface, preventing server memory overflow.
3.  **10-View Test-Time Augmentation (TTA)**: An advanced server-side inference pipeline that feeds 10 crops/flips of a user upload to the model, averaging softmax logits to eliminate framing noise.
4.  **Bilingual Taxonomy Lookup**: A client-side translation engine that parses `flower-names.txt` to translate scientific Latin names into Turkish and English common names on the fly.
5.  **Academic Slide Deck Generator**: A programmatic PowerPoint compiler (`generate_pptx.py`) that exports a 13-slide premium widescreen deck with built-in WebP-to-PNG image helpers.

---

## 2. Dataset Ecosystem: Pl@ntNet & Pl@ntNet-300K

### 2.1. The Pl@ntNet Platform
Pl@ntNet is a global citizen-science platform and visual database where millions of amateur naturalists, hikers, and botanists upload plant observations daily. Observations are validated and labeled by professional botanists, resulting in a massive, high-quality botanical inventory.

### 2.2. The Pl@ntNet-300K Benchmark Dataset (NeurIPS 2021)
Pl@ntNet-300K is a subset curated from Pl@ntNet observations, released as a formal benchmark at the NeurIPS 2021 conference. It contains **306,302 images** covering **1,081 species**.

```
                        Pl@ntNet-300K Dataset Splits
┌─────────────────────────────────┬─────────────────────────────────┬──────────┐
│ Train Split (243,917 images)    │ Validation Split (31,170 images)│ Test     │
│ 80%                             │ 10%                             │ 10%      │
└─────────────────────────────────┴─────────────────────────────────┴──────────┘
```

### 2.3. The Long-Tail Challenge
In nature, common species (e.g., daisies) are photographed thousands of times, while rare wild species are pictured only a handful of times. In Pl@ntNet-300K, **80% of species reside in just 11% of the images (Lorentz Curve distribution)**. The weights integrated into this project were trained using class-balanced loss functions to mitigate this imbalance.

---

## 3. Supported Model Architecture Library

We integrate 6 diverse architectures sourced from the official repository, evaluating their trade-offs:

1.  **Vision Transformer (ViT-Base-Patch16-224 - 693.1 MB)**: Discards convolutional layers completely. Splits images into 14x14 flat patches, applies Positional Encodings, and processes global relationships via Multi-Head Self-Attention (MSA).
2.  **ResNet-152 (483.9 MB)**: A very deep ResNet utilizing skip connections. Excellent at capturing local texture patterns and leaf venation.
3.  **ResNet-50 (206.1 MB)**: A standard residual network balancing inference latency and accuracy.
4.  **EfficientNet-B4 (156.8 MB)**: Leverages compound scaling to balance width, depth, and resolution for state-of-the-art parameter efficiency.
5.  **DenseNet-121 (65.2 MB)**: Connects all layers directly to reuse features and minimize parameters.
6.  **MobileNet V3 Small (21.2 MB)**: Designed via Hardware-Aware Neural Architecture Search (NAS) for real-time mobile and edge device inference.

---

## 4. Advanced Inference & Engineering Implementations

### 4.1. 10-View Test-Time Augmentation (TTA)
Real-world images uploaded by users are often off-center or poorly framed. To prevent erroneous predictions, our Flask API utilizes a parallel TTA pipeline during inference:

```
                            10-View TTA Pipeline Flow
                             ┌────────────────────────┐
                             │   User Uploaded Image  │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │      Resize 384px      │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │ 5-Crop (4 Corners+Ctr) │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │   x2 Horizontal Flip   │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │ 10 Parallel View Batch │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │  Softmax Logits Aver.  │
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │ Final Top-5 Prediction │
                             └────────────────────────┘
```

### 4.2. Out-of-Distribution (OOD) Warning
If the top-1 prediction confidence falls below **15.0%**, the platform triggers an interactive alert: *"Low confidence prediction. The uploaded image might not belong to the 1,081 cataloged species or the picture suffers from lighting/framing noise."*

---

## 5. Architectural Benchmarking & Trade-Offs

We evaluated these models across two challenging qualitative test cases:

### Case Study 1: Arbutus unedo (Dağ Çileği / Strawberry Tree)
*   **ViT Base Patch16**: Achieved **76.54%** top-1 confidence, parsing global leaf-fruit spatial relationships even when fruits were partially occluded.
*   **ResNet-152 & EfficientNet-B4**: Achieved **72.34%** and **74.12%** respectively, picking up local dentate margin patterns of the leaves.
*   **MobileNet V3**: Correctly classified the species in Rank 1 but at a lower confidence of **48.20%**.

### Case Study 2: Acacia xanthophloea (Ateş Ağacı / Fever Tree)
*   **CNNs (ResNet/DenseNet)**: Excelled at recognizing the distinct smooth, yellowish-green powdery bark textures of the trunk.
*   **ViT**: Excelled at capturing the global spatial sparse arrangement of the bipinnate leaves when the trunk was not in frame.

### Performance vs. Computational Footprint Trade-off
*   **Heavyweights (ViT, ResNet-152)**: Maximum accuracy, best for cloud servers, high latency on CPUs.
*   **Balanced (EfficientNet-B4, DenseNet-121)**: Ideal for high-throughput web APIs.
*   **Edge Champion (MobileNet V3)**: Compact size (21.2 MB), fits offline mobile files, ultra-fast CPU inference.

---

## 6. Project Structure

```
DLP4/
├── models/                           # Downloaded pre-trained weights (.tar)
├── templates/
│   └── index.html                    # Drag-and-drop bilingual web UI
├── static/
│   ├── css/                          # Modern dark-mode vanilla CSS
│   └── js/                           # Real-time AJAX and Charting JS
├── app.py                            # Flask server, TTA, Dynamic caching
├── generate_pptx.py                  # Automated 13-slide PowerPoint builder
├── blog_post.md                      # Academic Turkish write-up
├── flower-names.txt                  # Bilingual dictionary lookup (Latin-TR-EN)
├── .gitignore                        # Git exclusion rules
└── README.md                         # This file
```

---

## 7. Programmatic Presentation Slide Deck

The included `generate_pptx.py` compiled a highly-polished, widescreen (16:9) 13-slide presentation (`dl4sunumu_guncel.pptx`) with a custom forest-green corporate theme, container cards, and statistics metrics:

*   **Slide 1**: Title & Cover (Platform Overview)
*   **Slide 2**: Motivation & Problem Definition
*   **Slide 3**: Pl@ntNet Platform & Pl@ntNet-300K Dataset Curations
*   **Slide 4**: Class Imbalance & The Lorentz Curve Analytes
*   **Slide 5**: Pre-trained GitHub Weights Zoo & Our Core Contributions
*   **Slide 6**: Vision Transformer (ViT) Deep-Dive Architecture
*   **Slide 7**: Inference Pipeline & 10-View Test-Time Augmentation (TTA)
*   **Slide 8**: Flask Web Architecture & Dynamic Memory Caching
*   **Slide 9**: Qualitative Case Study 1: Strawberry Tree (*Arbutus unedo*)
*   **Slide 10**: Qualitative Case Study 2: Fever Tree (*Acacia xanthophloea*)
*   **Slide 11**: Model Zoo Comparative Report (Accuracy vs. Latency/Size)
*   **Slide 12**: Challenges, Closed-Set Limits & Visual Ambiguities
*   **Slide 13**: Outro, Future Extensions (ONNX, TFLite, GPS GIS filtering)

---

## 8. Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/halilakbas11/DL4-PlantNET300K-ClassifierAnalysis.git
    cd DL4-PlantNET300K-ClassifierAnalysis
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Needs: flask, torch, torchvision, timm, pillow, python-pptx
    ```

3.  **Place Pre-trained Weights**:
    Download the Pl@ntNet-300K trained weight tar archives and place them into the `models/` directory.

4.  **Run the Web Application**:
    ```bash
    python app.py
    ```
    Access the UI at `http://127.0.0.1:5000/`.

5.  **Compile the Slides (Optional)**:
    ```bash
    python generate_pptx.py
    ```
    This generates `dl4sunumu.pptx` (or `dl4sunumu_guncel.pptx` if PowerPoint is open).
