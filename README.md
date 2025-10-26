🧠 AI Agent-Based Deepfake Detection for Images

An AI-powered web application that detects deepfakes and manipulated images using a multi-agent forensic system integrated with EfficientNetV2-S deep learning architecture. The project performs advanced Error Level Analysis (ELA), Pixel & Frequency Artifact Detection, Noise Pattern Inspection, and Metadata Forensics, offering a comprehensive digital media authentication tool.

🚀 Overview

This project introduces an AI Agent Framework for deepfake detection and image forensics.
Each agent analyzes a specific forensic dimension—compression patterns, noise, frequency irregularities, pixel artifacts, and metadata—to collectively assess an image’s authenticity.

The final verdict is produced using a Decision Orchestrator, combining agent confidence scores and model predictions to classify the image as:

🟢 Likely Authentic | 🟡 Suspicious | 🔴 Manipulated/Fake

⚙️ Features
🔬 Multi-Agent Forensic Framework

ELA Agent: Detects inconsistent compression artifacts.

Pixel Artifact Agent: Highlights edge and texture abnormalities.

Frequency Domain Agent: Detects irregularities in frequency spectra.

Noise Pattern Agent: Analyzes unnatural noise variance.

Metadata Forensics Agent: Examines EXIF data and cryptographic hashes.

AI Model Agent: EfficientNetV2-S deep learning classifier (Real/Fake).

🤖 Deep Learning Model

Architecture: EfficientNetV2-S (Transfer Learning from ImageNet)

Training Accuracy: 98%

Validation Accuracy: 97%

Input Size: 256×256

Optimizer: AdamW with OneCycleLR scheduler

Loss Function: CrossEntropyLoss with label smoothing

Batch Size: 32

Epochs: 30 (Early stopping enabled)

💻 Web Application

Built with Streamlit

Upload any image (.jpg, .jpeg, .png)

View:

AI detection result with confidence score

ELA visualization

Metadata & Hash inspection

Interactive plots (Gauge + Agent Bar Chart)

Export results as JSON or HTML report

🧩 Tech Stack
Category	Tools & Frameworks
Frontend	Streamlit
Backend / ML	PyTorch, Torchvision
Image Processing	OpenCV, Pillow
Visualization	Plotly
Utility	NumPy, Pandas, hashlib, tqdm
📁 Project Structure
AI-Agent-Deepfake-Detection/
│
├── app.py                 # Streamlit web interface and multi-agent pipeline
├── train.py               # Model training with EfficientNetV2-S
├── size_check.py           # Dataset integrity and sample counting
├── requirements.txt        # Python dependencies
└── best_deepfake_v2s.pth   # Trained model weights (place in root directory)

🔧 Installation

Clone the Repository

git clone https://github.com/ShrutiKandagatla/AI-Agent-Deepfake-Detection.git
cd AI-Agent-Deepfake-Detection


Install Dependencies

pip install -r requirements.txt


Ensure Trained Model is Present
Place your trained model file:

best_deepfake_v2s.pth


in the project directory.
If unavailable, train your model using:

python train.py

🧠 Usage
Run the Streamlit Web App
streamlit run app.py


Then open your browser at:

http://localhost:8501

Upload an Image

View AI prediction with confidence level

See ELA comparison

Inspect metadata and hash values

Download full JSON/HTML reports

🧾 Training Details

Dataset Folder Structure:

Data Set 1/
├── train/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/


Training Command:

python train.py


Performance Metrics:

Metric	Accuracy
Training	98%
Validation	97%
📊 Output Example

Final Verdict: 🟢 Likely Authentic

Suspicion Score: 23.5%

Agents Summary:

ELA Agent → Low Risk

Pixel Agent → Natural Patterns

Frequency Agent → Natural Spectrum

Model Agent → 92% Real Confidence

🚨 Limitations

May misclassify low-resolution or heavily compressed images.

Trained primarily on static images (video deepfakes not yet supported).

Results depend on dataset diversity.

🔄 Future Enhancements

Video-level deepfake detection

Real-time webcam verification

Ensemble of CNN and Vision Transformer models

REST API for third-party integration

📄 License

This project is developed for educational and research purposes.
Please ensure compliance with applicable privacy and ethical guidelines.

👩‍💻 Maintainer

Shruti Kandagatla
📧 [Your Email]
🌐 GitHub: ShrutiKandagatla
