import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Squat Form AI Analyzer", page_icon="🏋️‍♂️", layout="wide")

st.title("🏋️‍♂️ AI Squat Form & Valgus Analyzer")
st.write("Upload a squat video to analyze knee valgus trajectory and confidence over time.")

# --- 1. LOAD ML MODEL & MEDIAPIPE ---
MODEL_PATH = "valgus_classifier.pkl"
MEDIAPIPE_MODEL_PATH = "pose_landmarker_full.task"
TRACKED_LANDMARKS = [23, 24, 25, 26, 27, 28]

feature_names = []
for lm_id in TRACKED_LANDMARKS:
    feature_names.extend([f"lm_{lm_id}_x", f"lm_{lm_id}_y", f"lm_{lm_id}_z"])

@st.cache_resource
def load_models():
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return ml_model, detector

if not os.path.exists(MODEL_PATH) or not os.path.exists(MEDIAPIPE_MODEL_PATH):
    st.error("❌ Model files missing. Ensure valgus_classifier.pkl and pose_landmarker_full.task are in the directory.")
    st.stop()

ml_model, detector = load_models()

# --- 2. FILE UPLOADER SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload a Squat Video", type=["mp4", "mov", "avi", "m4v"])

if uploaded_file is not None:
    # Save uploaded file to temp disk location for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Upload")
        st.video(tfile.name)

    if st.button("🚀 Analyze Squat Form"):
        with col2:
            st.subheader("Video Playback Stream")
            st_frame = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            
            frame_records = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                time_seconds = frame_count / fps
                timestamp_ms = int(time_seconds * 1000)

                # MediaPipe Inference
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)

                prediction_label = "No Pose"
                confidence = 0.0
                valgus_score = 0.0  # Normalized 0-100 scale for plotting

                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = detection_result.pose_landmarks[0]
                    features = []
                    for lm_id in TRACKED_LANDMARKS:
                        lm = landmarks[lm_id]
                        features.extend([lm.x, lm.y, lm.z])

                    X_frame = pd.DataFrame([features], columns=feature_names)
                    prediction_label = ml_model.predict(X_frame)[0]
                    probabilities = ml_model.predict_proba(X_frame)[0]
                    
                    valgus_idx = list(ml_model.classes_).index("Valgus") if "Valgus" in ml_model.classes_ else 1
                    valgus_score = probabilities[valgus_idx] * 100
                    
                    class_index = list(ml_model.classes_).index(prediction_label)
                    confidence = probabilities[class_index] * 100

                    frame_records.append({
                        "Frame": frame_count,
                        "Time (s)": round(time_seconds, 2),
                        "Form": prediction_label,
                        "Valgus Likelihood (%)": round(valgus_score, 2),
                        "Confidence (%)": round(confidence, 2)
                    })

                # Display CLEAN video frame (no text or boxes drawn on frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            st.success("Analysis Complete!")

            # --- 3. DASHBOARD METRICS & INTERACTIVE CHARTS ---
            if frame_records:
                df_results = pd.DataFrame(frame_records)

                st.markdown("---")
                st.header("📊 Performance & Alignment Analytics")

                # High-level Metrics Summary Cards
                m1, m2, m3 = st.columns(3)
                valgus_frames = (df_results["Form"] == "Valgus").sum()
                total_detected = len(df_results)
                overall_form = "Valgus Detected" if valgus_frames > (total_detected / 2) else "Clean Form"
                avg_confidence = df_results["Confidence (%)"].mean()

                m1.metric("Overall Assessed Form", overall_form)
                m2.metric("Average Model Confidence", f"{avg_confidence:.1f}%")
                m3.metric("Valgus Frames / Total", f"{valgus_frames} / {total_detected}")

                # Interactive Plotly Chart: Valgus Likelihood Over Time
                st.subheader("📈 Valgus Risk Trajectory Over Time")
                fig = px.line(
                    df_results, 
                    x="Time (s)", 
                    y="Valgus Likelihood (%)",
                    color="Form",
                    color_discrete_map={"Clean": "#00CC96", "Valgus": "#EF553B"},
                    title="Frame-by-Frame Knee Cave Probability",
                    hover_data=["Frame", "Confidence (%)"]
                )
                
                # Add a 50% decision threshold line
                fig.add_hline(
                    y=50, 
                    line_dash="dash", 
                    line_color="gray", 
                    annotation_text="Valgus Threshold (50%)", 
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Valgus Likelihood (%)",
                    yaxis_range=[0, 105],
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

# --- 4. FOOTER ---
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #888888; font-size: 14px;">
        Developed by <b>Your Name</b> | Contact: <a href="mailto:your.email@example.com">your.email@example.com</a>
    </div>
    """,
    unsafe_allow_html=True
)