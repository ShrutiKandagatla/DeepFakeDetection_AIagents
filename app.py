import streamlit as st
from PIL import Image, ImageChops, ExifTags
import hashlib
import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import datetime
import json
import cv2
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from typing import Any

# ==========================================================
# 1. Enhanced ELA Analysis Agent
# ==========================================================
def perform_ela(image, quality=90):
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, "JPEG", quality=quality)
    resaved = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = (255.0 / max_diff) * 2.5 if max_diff > 0 else 1
    ela_image = ela_image.point(lambda p: min(int(p * scale), 255))
    os.remove(temp_filename)
    return ela_image

def ela_agent(image):
    ela_img = perform_ela(image)
    ela_arr = np.array(ela_img)
    score = np.mean(ela_arr)
    std_dev = np.std(ela_arr)
    max_val = np.max(ela_arr)

    if score > 100:
        confidence = "High"; risk_level = "🔴 High Risk"
    elif score > 60:
        confidence = "Medium"; risk_level = "🟡 Medium Risk"
    else:
        confidence = "Low"; risk_level = "🟢 Low Risk"

    explanation = f"""
    **ELA Analysis Results:**
    - Avg pixel difference: {score:.2f}
    - Std deviation: {std_dev:.2f}
    - Max diff: {max_val}
    - Risk: {risk_level}
    """

    return {
        "agent_name": "ELA Compression Agent",
        "deepfake_suspect": score > 80,
        "confidence": confidence,
        "score": score,
        "explanation": explanation
    }

# ==========================================================
# 2. Pixel Artifact Agent
# ==========================================================
def pixel_artifact_agent(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_score = np.mean(np.abs(laplacian))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_score = np.mean(np.sqrt(sobelx**2 + sobely**2))
    edges = cv2.Canny(gray, 50, 150)
    canny_score = np.sum(edges > 0) / edges.size * 100
    combined_score = (lap_score + sobel_score / 10 + canny_score) / 3

    if combined_score > 40:
        confidence = "High"; risk_level = "🔴 Suspicious Artifacts"
    elif combined_score > 25:
        confidence = "Medium"; risk_level = "🟡 Some Artifacts"
    else:
        confidence = "Low"; risk_level = "🟢 Natural Patterns"

    return {
        "agent_name": "Pixel Artifact Agent",
        "deepfake_suspect": combined_score > 30,
        "confidence": confidence,
        "score": combined_score,
        "explanation": f"""
        **Pixel Artifact Analysis:**
        - Laplacian: {lap_score:.2f}
        - Sobel: {sobel_score:.2f}
        - Canny: {canny_score:.2f}
        - Risk: {risk_level}
        """
    }

# ==========================================================
# 3. Frequency Domain Agent
# ==========================================================
def frequency_domain_agent(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    mean_mag = np.mean(magnitude_spectrum)
    std_mag = np.std(magnitude_spectrum)
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    high_freq = magnitude_spectrum[center_h - h//4:center_h + h//4, center_w - w//4:center_w + w//4]
    low_freq = magnitude_spectrum[center_h - h//8:center_h + h//8, center_w - w//8:center_w + w//8]
    ratio = np.mean(high_freq) / (np.mean(low_freq) + 1e-6)
    anomaly = abs(mean_mag - 180) + abs(ratio - 0.7) * 50

    if anomaly > 60:
        confidence = "High"; risk = "🔴 Spectral Anomalies"
    elif anomaly > 35:
        confidence = "Medium"; risk = "🟡 Irregularities"
    else:
        confidence = "Low"; risk = "🟢 Natural Spectrum"

    return {
        "agent_name": "Frequency Domain Agent",
        "deepfake_suspect": anomaly > 45,
        "confidence": confidence,
        "score": anomaly,
        "explanation": f"""
        **Frequency Domain Analysis:**
        - Mean: {mean_mag:.2f}
        - Ratio: {ratio:.2f}
        - Risk: {risk}
        """
    }

# ==========================================================
# 4. Metadata Forensics Agent
# ==========================================================
def get_file_hashes(image_bytes):
    return (
        hashlib.md5(image_bytes).hexdigest(),
        hashlib.sha1(image_bytes).hexdigest(),
        hashlib.sha256(image_bytes).hexdigest()
    )

def extract_exif_data(image):
    exif_data = {}
    try:
        exif = image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
    except:
        pass
    return exif_data

def metadata_forensics_agent(image, filename, image_bytes):
    md5, sha1, sha256 = get_file_hashes(image_bytes)
    exif = extract_exif_data(image)
    file_size = len(image_bytes)
    suspicious = []
    score = 100

    if not exif:
        suspicious.append("No EXIF metadata found")
        score -= 30

    if score >= 80:
        conf, risk = "High", "🟢 Likely Authentic"
    elif score >= 50:
        conf, risk = "Medium", "🟡 Questionable"
    else:
        conf, risk = "High", "🔴 Likely Synthetic"

    return {
        "agent_name": "Metadata Forensics Agent",
        "deepfake_suspect": score < 60,
        "confidence": conf,
        "score": score,
        "explanation": f"""
        **Metadata Forensics:**
        - File: {filename}
        - Size: {file_size/1024:.2f} KB
        - Risk: {risk}
        """
    }

# ==========================================================
# 5. Noise Pattern Agent (FIXED Missing Agent)
# ==========================================================
def noise_metadata_agent(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    noise_map = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = np.var(noise_map)

    if noise_level > 500:
        conf = "High"; risk = "🔴 Unnatural Noise"
    elif noise_level > 200:
        conf = "Medium"; risk = "🟡 Some Noise Irregularities"
    else:
        conf = "Low"; risk = "🟢 Natural Noise"

    return {
        "agent_name": "Noise Pattern Agent",
        "deepfake_suspect": noise_level > 350,
        "confidence": conf,
        "score": noise_level,
        "explanation": f"""
        **Noise Analysis:**
        - Variance: {noise_level:.2f}
        - Risk: {risk}
        """
    }

# ==========================================================
# 6. EfficientNetV2-S Model Agent
# ==========================================================
def create_model():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model

@st.cache_resource
def load_model():
    try:
        model = create_model()
        state_dict = torch.load("best_deepfake_v2s.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None

def model_agent(image, model):
    if model is None:
        return {"agent_name": "AI Model Agent", "deepfake_suspect": False, "confidence": "None", "score": 0}

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        # ImageFolder orders classes alphabetically -> {'fake':0, 'real':1}
        # Model output index 0 = FAKE, index 1 = REAL
        fake_prob = float(probs[0, 0])

    risk = "🔴 Fake" if fake_prob > 0.8 else ("🟡 Possibly Fake" if fake_prob > 0.6 else "🟢 Real")

    return {
        "agent_name": "AI Model Agent",
        "deepfake_suspect": fake_prob > 0.5,
        "confidence": "High" if fake_prob > 0.7 else "Medium",
        "score": fake_prob * 100,
        "explanation": f"Fake Probability: {fake_prob*100:.2f}% → {risk}"
    }

# ==========================================================
# 7. Decision Orchestrator
# ==========================================================
def decision_orchestrator(results):
    weights = {"Very High": 1.0, "High": 0.8, "Medium": 0.6, "Low": 0.3}
    total_weight, weighted_score = 0, 0
    for r in results:
        w = weights.get(r.get("confidence", "Low"), 0.3)
        weighted_score += w * (1 if r.get("deepfake_suspect", False) else 0)
        total_weight += w
    ratio = weighted_score / total_weight if total_weight else 0

    if ratio > 0.7: verdict = "🔴 FAKE/MANIPULATED"
    elif ratio > 0.4: verdict = "🟡 SUSPICIOUS"
    else: verdict = "🟢 LIKELY AUTHENTIC"

    return {"final_verdict": verdict, "suspicion_ratio": ratio, "results": results}

# ==========================================================
# 8. Helpers: JSON serialization, charts, metadata
# ==========================================================
def to_jsonable(obj: Any):
    """Recursively convert numpy and other non-serializable types to JSON-safe Python types."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Numpy scalars/arrays
    if _np is not None:
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    # Bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8', errors='ignore')
        except Exception:
            return str(obj)

    # Datetime
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # Mapping
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Iterable
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # Fallback
    return str(obj)

def render_gauge(ratio: float):
    value = max(0.0, min(1.0, float(ratio))) * 100
    color = '#2ecc71' if value < 40 else ('#f1c40f' if value < 70 else '#e74c3c')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': '%'},
        title={'text': 'Suspicion Score'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': '#e8f5e9'},
                {'range': [40, 70], 'color': '#fff8e1'},
                {'range': [70, 100], 'color': '#ffebee'}
            ]
        }
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def render_agent_bar(results):
    names = [r.get('agent_name', 'Agent') for r in results]
    scores = [float(r.get('score', 0)) for r in results]
    fig = go.Figure(go.Bar(x=names, y=scores, marker_color='#3498db'))
    fig.update_layout(
        title='Agent Scores',
        xaxis_title='Agent', yaxis_title='Score',
        height=300, margin=dict(l=10, r=10, t=30, b=60)
    )
    return fig

def basic_image_metadata(image: Image.Image, filename: str, image_bytes: bytes):
    w, h = image.size
    size_kb = len(image_bytes) / 1024.0
    mpix = (w * h) / 1_000_000
    md5, sha1, sha256 = get_file_hashes(image_bytes)
    return {
        'filename': filename,
        'dimensions': f'{w}x{h}',
        'megapixels': round(mpix, 3),
        'format': image.format or 'RGB',
        'mode': image.mode,
        'size_kb': round(size_kb, 2),
        'hash_md5': md5,
        'hash_sha1': sha1,
        'hash_sha256': sha256,
    }

def build_html_report(final_dict: dict, metadata: dict):
    verdict = final_dict.get('final_verdict', 'N/A')
    ratio = float(final_dict.get('suspicion_ratio', 0)) * 100
    rows = ''
    for r in final_dict.get('results', []):
        rows += f"<tr><td>{r.get('agent_name','')}</td><td>{r.get('confidence','')}</td><td>{r.get('score','')}</td><td>{r.get('deepfake_suspect','')}</td></tr>"
    meta_rows = ''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in metadata.items()])
    html = f"""
    <html><head><meta charset='utf-8'><title>Deepfake Report</title>
    <style>body{{font-family:Arial, sans-serif;padding:16px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:8px}}</style>
    </head><body>
    <h2>Deepfake Detection Report</h2>
    <p><strong>Verdict:</strong> {verdict}</p>
    <p><strong>Suspicion Score:</strong> {ratio:.2f}%</p>
    <h3>Image Metadata</h3>
    <table>{meta_rows}</table>
    <h3>Agents</h3>
    <table><tr><th>Agent</th><th>Confidence</th><th>Score</th><th>Suspect</th></tr>{rows}</table>
    <p>Generated by Deepfake Detection Dashboard</p>
    </body></html>
    """
    return html

# ==========================================================
# 8. Streamlit Main App
# ==========================================================
def main():
    st.set_page_config(page_title="Deepfake Detection AI", layout="wide")
    st.title("🧠 Deepfake Detection and Media Forensics Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Project")
        st.markdown("DeepFake Detection & Media Auth | EfficientNetV2-S + Forensic Agents")
        st.divider()
        st.header("Controls")
        run_ela = st.checkbox("ELA Agent", value=True)
        run_pix = st.checkbox("Pixel Artifact Agent", value=True)
        run_freq = st.checkbox("Frequency Domain Agent", value=True)
        run_meta = st.checkbox("Metadata Agent", value=True)
        run_noise = st.checkbox("Noise Agent", value=True)
        run_model = st.checkbox("AI Model Agent", value=True)
        st.divider()
        st.caption("Tip: Disable agents to speed up on large images.")

    uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"])
    model = load_model()

    if uploaded_file:
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        filename = uploaded_file.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Original Image", use_column_width=True)
        with col2:
            st.image(perform_ela(image), caption="🔬 ELA Analysis", use_column_width=True)

        st.subheader("🤖 Running Multi-Agent Deepfake Analysis...")

        results = []
        if run_ela:
            results.append(ela_agent(image))
        if run_pix:
            results.append(pixel_artifact_agent(image_cv))
        if run_freq:
            results.append(frequency_domain_agent(image_cv))
        if run_meta:
            results.append(metadata_forensics_agent(image, filename, image_bytes))
        if run_noise:
            results.append(noise_metadata_agent(image_cv))
        if run_model:
            results.append(model_agent(image, model))

        final = decision_orchestrator(results)
        tabs = st.tabs(["Overview", "Agents", "Model", "Image Info", "Export"])

        with tabs[0]:
            st.markdown(f"### 🧾 Final Verdict: {final['final_verdict']}")
            st.plotly_chart(render_gauge(final["suspicion_ratio"]), use_container_width=True)
            st.plotly_chart(render_agent_bar(results), use_container_width=True)

        with tabs[1]:
            st.markdown("### 🔍 Agent Explanations")
            for r in results:
                with st.expander(f"{r.get('agent_name','Agent')} — {r.get('confidence','')}"):
                    st.write(f"Score: {r.get('score','N/A')}")
                    st.info(r.get('explanation', ''))

        with tabs[2]:
            st.markdown("### 🤖 AI Model Output")
            if run_model:
                ma = [x for x in results if x.get('agent_name') == 'AI Model Agent']
                if ma:
                    st.write(ma[0].get('explanation', ''))
            else:
                st.info("Model agent disabled in the sidebar.")

        with tabs[3]:
            st.markdown("### 🖼️ Image Metadata")
            meta = basic_image_metadata(image, filename, image_bytes)
            c1, c2 = st.columns(2)
            with c1:
                st.write({k: meta[k] for k in ['filename','dimensions','megapixels','format','mode']})
            with c2:
                st.write({k: meta[k] for k in ['size_kb','hash_md5','hash_sha1','hash_sha256']})

        with tabs[4]:
            st.markdown("### ⬇️ Export")
            # JSON
            final_json = json.dumps(to_jsonable(final), indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Download Results (JSON)",
                data=final_json,
                file_name=f"{filename}_results.json",
                mime="application/json"
            )
            # HTML report
            meta = basic_image_metadata(image, filename, image_bytes)
            html_report = build_html_report(to_jsonable(final), meta)
            st.download_button(
                "📄 Download Report (HTML)",
                data=html_report,
                file_name=f"{filename}_report.html",
                mime="text/html"
            )
    else:
        st.info("👆 Upload an image to start analysis.")

if __name__ == "__main__":
    main()
