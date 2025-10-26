# 🧠 AI Agent-Based Deepfake Detection for Images

An **AI-powered web application** that detects deepfakes and manipulated images using a **multi-agent forensic system** integrated with **EfficientNetV2-S** deep learning architecture.  
The system performs **Error Level Analysis (ELA)**, **Pixel & Frequency Artifact Detection**, **Noise Pattern Inspection**, and **Metadata Forensics**, offering a comprehensive solution for image authenticity verification.

---

## 🚀 Overview

This project introduces an **AI Agent Framework** for deepfake detection and image forensics.  
Each agent specializes in a unique forensic dimension—compression patterns, frequency irregularities, pixel artifacts, noise, and metadata—to collaboratively determine the authenticity of an image.

The **Decision Orchestrator** fuses all agent outputs and deep learning predictions to produce a final verdict:

> 🟢 Likely Authentic | 🟡 Suspicious | 🔴 Manipulated/Fake

---

## ⚙️ Features

### 🔬 Multi-Agent Forensic Framework
- **ELA Agent:** Detects inconsistent compression artifacts.
- **Pixel Artifact Agent:** Identifies edge and texture irregularities.
- **Frequency Domain Agent:** Highlights frequency spectrum anomalies.
- **Noise Pattern Agent:** Measures unnatural noise variance.
- **Metadata Forensics Agent:** Inspects EXIF data and computes cryptographic hashes.
- **AI Model Agent:** EfficientNetV2-S deep learning classifier (Real/Fake).

### 🤖 Deep Learning Model
- **Architecture:** EfficientNetV2-S (Transfer Learning from ImageNet)
- **Training Accuracy:** 98%
- **Validation Accuracy:** 97%
- **Input Size:** 256×256 pixels
- **Optimizer:** AdamW with OneCycleLR scheduler  
- **Loss Function:** CrossEntropyLoss with label smoothing  
- **Batch Size:** 32  
- **Epochs:** 30 (Early stopping enabled)

### 💻 Web Application
- Built with **Streamlit**
- Upload any image (`.jpg`, `.jpeg`, `.png`)
- View:
  - AI prediction and confidence score  
  - ELA visualization  
  - Metadata and cryptographic hashes  
  - Interactive gauge and bar charts  
- Export results as **JSON** or **HTML Report**

---

## 🧩 Tech Stack

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Frontend** | Streamlit |
| **Backend / ML** | PyTorch, Torchvision |
| **Image Processing** | OpenCV, Pillow |
| **Visualization** | Plotly |
| **Utilities** | NumPy, Pandas, hashlib, tqdm |

---

## 📁 Project Structure

```text
AI-Agent-Deepfake-Detection/
│
├── app.py                 - Streamlit web interface and multi-agent pipeline
├── train.py               - Model training with EfficientNetV2-S
├── size_check.py          - Dataset integrity and sample counting
├── requirements.txt       - Python dependencies
└── best_deepfake_v2s.pth  - Trained model weights (place in root directory)
```

---

## 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ShrutiKandagatla/AI-Agent-Deepfake-Detection.git
   cd AI-Agent-Deepfake-Detection
   ```
2. **Install Dependecies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure Trained Model is Present**
   - Place your trained model file:
   ```bash
   best_deepfake_v2s.pth
   ```
   - in the project directory.
   - If unavailable, train your model using:
   ```bash
   python train.py
   ```
   
---

## 🧠 Usage

### Run the Streamlit Web App
```bash
streamlit run app.py
```

Then open your browser at:

> [http://localhost:8501](http://localhost:8501)

### Upload an Image

* View AI prediction with confidence level
* See ELA comparison
* Inspect metadata and hash values
* Download full **JSON/HTML** reports

---

## 🧾 **Training Details**

* **Dataset Folder Structure:**

```test
Data Set 1/
├── train/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

* **Training Command:**

```bash
python train.py
```

* **Performance Metrics:**

| Metric     | Accuracy |
| ---------- | -------- |
| Training   | 98%      |
| Validation | 97%      |

---

## 📊 **Output Example**

* **Final Verdict:** 🟢 Likely Authentic  
* **Suspicion Score:** 23.5%
* **Agents Summary:**
* ELA Agent → Low Risk  
* Pixel Agent → Natural Patterns  
* Frequency Agent → Natural Spectrum  
* Model Agent → 92% Real Confidence  

---

## 🚨 **Limitations**

* May misclassify low-resolution or heavily compressed images.
* Trained primarily on static images (video deepfakes not yet supported).
* Results depend on dataset diversity.

---

## 🔄 **Future Enhancements**

* Video-level deepfake detection  
* Real-time webcam verification  
* Ensemble of CNN and Vision Transformer models  
* REST API for third-party integration  

---

## 📄 **License**

This project is developed for **educational and research purposes**.  
Please ensure compliance with applicable privacy and ethical guidelines.

---

## 👩‍💻 **Maintainer**

**Shruti Kandagatla**  
📧 *shrple@yahoo.com*  
🌐 [GitHub: ShrutiKandagatla](https://github.com/ShrutiKandagatla)
