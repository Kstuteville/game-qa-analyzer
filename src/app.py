"""
streamlit_app.py
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


def main():
    st.title("🎮 Gameplay UX Analyzer")

    st.markdown("""
    Upload gameplay + face video → AI analyzes → You get a UX/QA insight report.
    """)

    if not MODULES_AVAILABLE:
        st.error("Some modules failed to load:")
        for err in IMPORT_ERRORS:
            st.warning(err)
        st.stop()

    st.divider()

    # ---------------------------
    # Sidebar
    # ---------------------------
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
        - `facenet-pytorch` + `mtcnn` (FER backend)
        - `opencv-python`
        - `numpy`
        - `torch` (CPU on Mac, GPU-enabled on Dell)
        - `ffmpeg-python` (video frame extraction)
        - `openai` (UX LLM reports)
        - `streamlit`

        **Environment Variables:**
        - `OPENAI_API_KEY` must be set before running
        """)

        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes **gameplay events** + **player emotions**  
        to generate an AI-based UX / QA insight report.
        """)

    # ---------------------------
    # Upload section
    # ---------------------------
    st.header("📤 Step 1 — Upload Videos")

    col1, col2 = st.columns(2)

    with col1:
        gp_file = st.file_uploader("Gameplay Video", type=["mp4", "mov", "avi"])

    with col2:
        face_file = st.file_uploader("Face Video", type=["mp4", "mov", "avi"])

    ready = gp_file and face_file

    st.divider()

    # ---------------------------
    # Run button
    # ---------------------------
    if st.button("🚀 Run Analysis", disabled=not ready, use_container_width=True):
        folder = setup_data_folder()

        # Save uploaded files
        save_uploaded_file(gp_file, folder / "gameplay_input.mp4")
        save_uploaded_file(face_file, folder / "face_input.mp4")

        merged, report = run_pipeline(folder, context)

        if merged and report:
            st.success("🎉 Analysis complete!")
            st.balloons()

            tab1, tab2, tab3 = st.tabs(["📄 UX Report", "📋 Timeline", "📈 Stats"])

            with tab1:
                st.subheader("AI UX Report")
                st.write(report)
                st.download_button(
                    "Download Report",
                    data=report,
                    file_name="ux_report.txt"
                )

            with tab2:
                st.subheader("Merged Timeline (first 10)")
                st.json(merged[:10])
                st.download_button(
                    "Download Full Timeline",
                    data=json.dumps(merged, indent=2),
                    file_name="merged_log.json"
                )

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


if __name__ == "__main__":
    main()
