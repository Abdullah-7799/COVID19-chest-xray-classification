import streamlit as st
import pandas as pd
from src.engine import InferenceEngine

# Configuration
WEIGHTS_PATH = "models/best_modelval_acc0.900331_val_loss0.269495.h5"

@st.cache_resource
def load_engine():
    return InferenceEngine(WEIGHTS_PATH)

engine = load_engine()

st.set_page_config(page_title="COVID-19 Diagnostic AI", layout="wide")
st.title("🩺 Chest X-Ray Classification System")
st.write("Professional Tool for Single & Batch COVID-19 Detection")

tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Processing"])

# --- Single Prediction ---
with tab1:
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], key="single")
    if uploaded_file:
        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            result = engine.predict(uploaded_file)
            st.subheader(f"Result: {result['label']}")
            st.progress(result['confidence'])
            st.write(f"Confidence: {result['confidence']:.2%}")
            
            # Show all probabilities
            st.bar_chart(result['scores'])

# --- Batch Prediction ---
with tab2:
    uploaded_files = st.file_uploader("Upload multiple images for batch processing", 
                                      type=["png", "jpg", "jpeg"], 
                                      accept_multiple_files=True)
    if uploaded_files:
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            res = engine.predict(file)
            batch_results.append({
                "Filename": file.name,
                "Prediction": res['label'],
                "Confidence": f"{res['confidence']:.2%}"
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        df = pd.DataFrame(batch_results)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Results as CSV", df.to_csv(index=False), "results.csv")