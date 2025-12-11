"""
Gameplay UX Analyzer + YOLO Fine-Tuner
Safe Mac + Dell version
"""

import streamlit as st
from pathlib import Path
import traceback
import json
import os

# Try importing modules without crashing the app
MODULES_AVAILABLE = True
IMPORT_ERRORS = []

try:
    from yolo_analyzer import analyze_gameplay
except Exception as e:
    MODULES_AVAILABLE = False
    IMPORT_ERRORS.append(f"YOLO analyzer error: {e}")

try:
    from fer_analyzer import analyze_face
except Exception as e:
    MODULES_AVAILABLE = False
    IMPORT_ERRORS.append(f"FER analyzer error: {e}")

try:
    from merge_logs import merge_logs
except Exception as e:
    MODULES_AVAILABLE = False
    IMPORT_ERRORS.append(f"Merge logs error: {e}")

try:
    from llm_agent import generate_ux_report
except Exception as e:
    MODULES_AVAILABLE = False
    IMPORT_ERRORS.append(f"LLM agent error: {e}")


# Page config
st.set_page_config(
    page_title="Gameplay UX Analyzer",
    page_icon="🎮",
    layout="wide"
)


# -------------------------
# Utility
# -------------------------

def setup_data_folder():
    folder = Path("data")
    folder.mkdir(exist_ok=True)
    return folder


def save_uploaded_file(uploaded, dest: Path):
    try:
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def run_pipeline(data_folder: Path, game_context: str):
    gp_video = data_folder / "gameplay_input.mp4"
    face_video = data_folder / "face_input.mp4"

    gameplay_log = data_folder / "gameplay_log.json"
    emotion_log = data_folder / "emotion_log.json"
    merged_path = data_folder / "merged_log.json"

    try:
        # Step 1 — YOLO
        with st.spinner("🎮 Running YOLO gameplay analysis..."):
            analyze_gameplay(str(gp_video), str(gameplay_log))

        # Step 2 — FER
        with st.spinner("😊 Running facial emotion analysis..."):
            analyze_face(str(face_video), str(emotion_log))

        # Step 3 — Merge logs
        with st.spinner("🔗 Merging gameplay + emotion timelines..."):
            merged = merge_logs(
                str(gameplay_log),
                str(emotion_log),
                str(merged_path),
                max_time_diff=2.0,
            )

        # Step 4 — LLM UX report
        with st.spinner("🤖 Generating UX insights..."):
            report = generate_ux_report(
                merged_log_path=str(merged_path),
                game_context=game_context
            )

        return merged, report

    except Exception as e:
        st.error("❌ Pipeline failed")
        st.code(traceback.format_exc())
        return None, None


# -------------------------
# UI
# -------------------------

def main():
    st.title("🎮 Gameplay UX Analyzer")

    st.markdown("""
    Upload gameplay + face video → AI analyzes → You get a UX/QA insight report.
    """)

    # Stop if any imports failed
    if not MODULES_AVAILABLE:
        st.error("Some modules failed to load:")
        for err in IMPORT_ERRORS:
            st.warning(err)
        st.stop()

    st.divider()

    # ----------------------------
    # Sidebar
    # ----------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        context = st.text_area(
            "Game Context",
            "Boss fight demo in action game",
            height=100
        )

        st.header("📦 Requirements")
        st.markdown("""
        **Core Dependencies:**
        - `ultralytics` (YOLOv8 gameplay analysis)
        - `fer` (facial emotion recognition)
        - `facenet-pytorch` + `mtcnn`
        - `opencv-python`, `numpy`
        - `torch` (CPU → Mac, GPU → Dell)
        - `openai`
        - `streamlit`

        **Environment Variable Required:**
        - `OPENAI_API_KEY`
        """)

        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes **gameplay events** + **player emotions**  
        to generate an AI-based UX / QA insight report.
        """)

    # ----------------------------
    # Upload section
    # ----------------------------
    st.header("📤 Step 1 — Upload Videos")

    col1, col2 = st.columns(2)
    with col1:
        gp_file = st.file_uploader("Gameplay Video", type=["mp4", "mov", "avi"])
    with col2:
        face_file = st.file_uploader("Face Video", type=["mp4", "mov", "avi"])

    ready = gp_file and face_file

    st.divider()

    if st.button("🚀 Run Analysis", disabled=not ready, use_container_width=True):
        folder = setup_data_folder()

        save_uploaded_file(gp_file, folder / "gameplay_input.mp4")
        save_uploaded_file(face_file, folder / "face_input.mp4")

        merged, report = run_pipeline(folder, context)

        if merged and report:
            st.success("🎉 Analysis complete!")
            st.balloons()

            tab1, tab2, tab3, tab4 = st.tabs([
                "📄 UX Report",
                "📋 Timeline",
                "📈 Stats",
                "🛠 Fine-Tune YOLO Detector"
            ])

            # -------------------------
            # Tab 1 — UX Report
            # -------------------------
            with tab1:
                st.subheader("AI UX Report")
                st.write(report)
                st.download_button("Download Report", data=report, file_name="ux_report.txt")

            # -------------------------
            # Tab 2 — Timeline
            # -------------------------
            with tab2:
                st.subheader("Merged Timeline (first 10)")
                st.json(merged[:10])
                st.download_button(
                    "Download Full Timeline",
                    data=json.dumps(merged, indent=2),
                    file_name="merged_log.json"
                )

            # -------------------------
            # Tab 3 — Stats
            # -------------------------
            with tab3:
                st.subheader("Event + Emotion Stats")

                event_counts = {}
                emotion_counts = {}

                for entry in merged:
                    event = entry.get("game_event", "unknown")
                    emo = entry.get("emotion_state", "unknown")

                    event_counts[event] = event_counts.get(event, 0) + 1
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

                st.write("### Events")
                st.json(event_counts)

                st.write("### Emotions")
                st.json(emotion_counts)

            # -------------------------
            # Tab 4 — YOLO Fine-Tuning
            # -------------------------
            with tab4:
                st.header("🛠 Fine-Tune YOLOv8 Detector")
                st.write("""
                Upload labeled gameplay frames to train a **custom YOLOv8 model**.
                This makes event detection MUCH more accurate for your game.
                """)

                from finetuner import extract_zip, create_dataset_yaml, train_yolov8

                dataset_zip = st.file_uploader("Upload YOLO Dataset (.zip)", type=["zip"])
                
                colA, colB = st.columns(2)
                with colA:
                    model_choice = st.selectbox("Base Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
                    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
                    imgsz = st.number_input("Image Size", min_value=320, max_value=1280, value=640)

                with colB:
                    st.info("""
                    **Dataset structure must be:**

                    ```
                    images/train
                    images/val
                    labels/train
                    labels/val
                    ```
                    """)

                if dataset_zip:
                    DATA_ROOT = Path("data/training")
                    DATA_ROOT.mkdir(parents=True, exist_ok=True)

                    zip_path = DATA_ROOT / "dataset.zip"
                    with open(zip_path, "wb") as f:
                        f.write(dataset_zip.getbuffer())

                    st.success("📦 Dataset uploaded")

                    if st.button("🚀 Start Fine-Tuning", use_container_width=True):
                        with st.spinner("Extracting dataset..."):
                            extracted = extract_zip(str(zip_path), str(DATA_ROOT))

                        with st.spinner("Generating YAML..."):
                            yaml_path = create_dataset_yaml(extracted)

                        with st.spinner("Training YOLOv8..."):
                            results = train_yolov8(
                                model_choice,
                                str(yaml_path),
                                epochs,
                                imgsz,
                                save_path="models/custom_yolov8.pt"
                            )

                        st.success("🎉 Training complete! Model saved to models/custom_yolov8.pt")

                        # Offer download
                        st.download_button(
                            "📥 Download Model",
                            data=open("models/custom_yolov8.pt", "rb").read(),
                            file_name="custom_yolov8.pt"
                        )


if __name__ == "__main__":
    main()
