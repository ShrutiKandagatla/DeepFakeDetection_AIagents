# 🔍 AI Agent-Based Deepfake Detection for Images

An AI-powered web application that detects deepfakes and manipulated images using EfficientNetV2 deep learning architecture combined with Error Level Analysis (ELA) and metadata inspection.

## 📊 Model Performance
- **Training Accuracy:** 98%
- **Validation Accuracy:** 97%
- **Architecture:** EfficientNetV2-S with transfer learning

## 🚀 Features

### 🤖 AI Detection
- **EfficientNetV2-S** deep learning model for binary classification (Real/Fake)
- Confidence scoring with detailed probability analysis
- Multi-level confidence categorization (Very High, High, Moderate, Low)

### 🔬 Forensic Analysis
- **Error Level Analysis (ELA)** to highlight potential manipulation areas
- **Metadata Inspection** for technical image properties
- **Cryptographic Hashing** (MD5, SHA1, SHA256) for integrity verification

### 📱 User Interface
- Clean, intuitive Streamlit web interface
- Side-by-side image comparison (Original vs ELA)
- Detailed results with confidence metrics
- Comprehensive image properties display

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the trained model:**
   - Make sure `best_deepfake_v2s.pth` is in the project directory
   - If not available, train the model first using `train.py`

4. **Test the setup (optional):**
   ```bash
   python test_app.py
   ```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the displayed URL (usually `http://localhost:8501`)

3. **Upload an image** using the file uploader

4. **View results:**
   - AI prediction with confidence score
   - ELA analysis visualization
   - Detailed image properties and metadata
   - Cryptographic hashes for verification

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── train.py                  # Model training script
├── test_app.py              # System test script
├── requirements.txt         # Python dependencies
├── best_deepfake_v2s.pth   # Trained model weights
└── Data Set 1-4/           # Training datasets
```

## 🔧 Technical Details

### Model Architecture
- **Base Model:** EfficientNetV2-S (pre-trained on ImageNet)
- **Modification:** Final layer replaced for binary classification
- **Input Size:** 224x224 pixels
- **Classes:** 0=Fake, 1=Real

### Image Processing
- **Preprocessing:** Resize, normalization with ImageNet stats
- **ELA Quality:** 90% JPEG compression for analysis
- **Supported Formats:** JPG, JPEG, PNG, BMP, TIFF

### Performance Metrics
- **Training Dataset:** Multiple datasets with real/fake splits
- **Batch Size:** 32
- **Optimizer:** Adam with learning rate 1e-4
- **Training Epochs:** 10 with early stopping

## 🎨 Features in Detail

### Confidence Analysis
- **Very High (≥85%):** Strong prediction confidence
- **High (≥70%):** Good prediction confidence  
- **Moderate (≥55%):** Reasonable prediction confidence
- **Low (<55%):** Uncertain prediction - manual verification recommended

### ELA Analysis
Error Level Analysis reveals:
- Areas with different compression levels
- Potential editing artifacts
- Inconsistencies in image quality
- Signs of digital manipulation

### Metadata Inspection
Examines:
- File format and compression details
- Image dimensions and color properties
- Unique color count and bit depth
- File size and technical specifications

## 🚨 Limitations

- **Model Performance:** 98% validation accuracy 
- **Dataset Dependency:** Performance varies with image types not seen in training
- **False Positives/Negatives:** Some authentic images may be flagged as fake and vice versa
- **Processing Limitations:** Best results with clear, high-quality images

## 🔄 Future Improvements

- [ ] Ensemble methods with multiple models
- [ ] Additional forensic techniques (noise analysis, lighting consistency)
- [ ] Video deepfake detection capabilities
- [ ] Real-time processing optimization
- [ ] Expanded training datasets

## 📄 License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using for content verification.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the detection capabilities and user experience.
