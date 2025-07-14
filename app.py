import streamlit as st
from core.training import train_model
from core.predictor import predict_disease, log_prediction
from utils.ocr_utils import extract_text_from_image, extract_text_from_pdf

# App setup
st.set_page_config(page_title="Medical Report Analyzer", layout="centered")
st.title("🧠 Medical Report Analyzer")
st.markdown("Upload or paste your medical report to get **diagnosis** and **medical recommendations**.")

# Input options
upload_type = st.radio("Choose input type", ["📝 Text", "📄 PDF", "🖼️ Image"])
report_text = ""

# Handle input
if upload_type == "📝 Text":
    report_text = st.text_area("Paste your medical report below:", height=200)

elif upload_type == "📄 PDF":
    pdf_file = st.file_uploader("Upload your report (PDF only)", type=["pdf"])
    if pdf_file:
        report_text = extract_text_from_pdf(pdf_file)

elif upload_type == "🖼️ Image":
    image_file = st.file_uploader("Upload your report image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if image_file:
        report_text = extract_text_from_image(image_file)

# Show extracted/pasted report
if report_text:
    st.markdown("### 🧾 Extracted Report Text:")
    st.code(report_text[:800])  # Show up to 800 characters

# Analyze Button
if st.button("🔍 Analyze Report", key="analyze_btn"):
    if report_text.strip():
        try:
            disease, tips = predict_disease(report_text)
            log_prediction(report_text, disease)
            st.success(f"🧬 **Predicted Disease:** {disease}")
            st.markdown("### 🩺 Recommendations:")
            for tip in tips:
                st.markdown(f"- {tip}")
        except Exception as e:
            st.error(f"❌ Error analyzing report: {e}")
    else:
        st.warning("⚠️ No report text found.")

# Retrain Button
if st.button("🔁 Retrain Model", key="retrain_btn"):
    try:
        train_model()
        st.success("✅ Model retrained successfully!")
    except Exception as e:
        st.error(f"❌ Model retraining failed: {e}")
