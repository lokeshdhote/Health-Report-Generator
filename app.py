import streamlit as st
from core.training import train_model
from core.predictor import predict_disease, log_prediction
from utils.ocr_utils import extract_text_from_image, extract_text_from_pdf
import traceback

# --- App setup ---
st.set_page_config(page_title="Medical Report Analyzer", layout="centered")
st.title("🧠 Medical Report Analyzer")
st.markdown("Upload or paste your medical report to get **diagnosis** and **medical recommendations**.")

# --- Input options ---
upload_type = st.radio("Choose input type", ["📝 Text", "📄 PDF", "🖼️ Image"])
report_text = ""

# --- Handle input ---
if upload_type == "📝 Text":
    report_text = st.text_area("Paste your medical report below:", height=200)

elif upload_type == "📄 PDF":
    pdf_file = st.file_uploader("Upload your report (PDF only)", type=["pdf"])
    if pdf_file:
        try:
            report_text = extract_text_from_pdf(pdf_file)
        except Exception as e:
            st.error(f"❌ PDF extraction failed: {e}")

elif upload_type == "🖼️ Image":
    image_file = st.file_uploader("Upload your report image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if image_file:
        try:
            report_text = extract_text_from_image(image_file)
        except Exception as e:
            st.error(f"❌ Image extraction failed: {e}")

# --- Show extracted/pasted report ---
if report_text:
    with st.expander("🧾 View full report"):
        st.code(report_text)

# --- Analyze Button ---
if st.button("🔍 Analyze Report", key="analyze_btn"):
    if report_text.strip():
        try:
            disease, tips = predict_disease(report_text)
            
            # Log prediction separately to avoid blocking analysis
            try:
                log_prediction(report_text, disease)
            except Exception as log_err:
                st.warning(f"⚠️ Logging failed: {log_err}")

            # Show results
            st.success(f"🧬 **Predicted Disease:** {disease}")
            st.markdown("### 🩺 Recommendations:")
            for tip in tips:
                st.markdown(f"- {tip}")

            # Download option
            result_text = f"Disease: {disease}\n\nRecommendations:\n" + "\n".join(tips)
            st.download_button("⬇️ Download Result", result_text, file_name="prediction.txt")
            
        except Exception as e:
            st.error(f"❌ Error analyzing report: {e}")
            # Log full traceback to file for debugging
            with open("error_log.txt", "a") as f:
                f.write(traceback.format_exc() + "\n\n")
    else:
        st.warning("⚠️ No report text found.")

# --- Retrain Button ---
if st.button("🔁 Retrain Model", key="retrain_btn"):
    try:
        with st.spinner("Retraining model... This may take a while."):
            train_model()
        st.success("✅ Model retrained successfully!")
    except Exception as e:
        st.error(f"❌ Model retraining failed: {e}")
        with open("error_log.txt", "a") as f:
            f.write(traceback.format_exc() + "\n\n")
