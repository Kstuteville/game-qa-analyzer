"""
Gameplay UX Analyzer + YOLO Fine-Tuner
Safe Mac + Dell version
"""

import streamlit as st
from pathlib import Path
import traceback
import json
import os

# ---------------------------------------------------------
# MODULE IMPORTS (safe wrappers)
# ---------------------------------------------------------

ANALYZER_OK = True
IMPORT_ERRORS = []

try:
    from yolo_analyzer import analyze_gameplay
except Exception as e:
    ANALYZER_OK = False
    IMPORT_ERRORS.append(f"YOLO analyzer error: {e}")

try:
    from fer_analyzer import analyze_face
except Exception as e:
    ANALYZER_OK = False
    IMPORT_ERRORS.append(f"Emotion analyzer (FER) error: {e}")

try:
    from merge_logs import merge_logs
except Exception as e:
    ANALYZER_OK = False
    IMPORT_ERRORS.append(f"Merge logs error: {e}")

try:
    from llm_agent import generate_ux_report
except Exception as e:
    ANALYZER_OK = False
    IMPORT_ERRORS.append(f"LLM agent error: {e}")

# YOLO FINE-TUNER MODULE IS OPTIONAL
FINETUNER_OK = True
try:
    from finetuner import extract_zip, create_dataset_yaml, train_yolov8
except Exception as e:
    FINETUNER_OK = False
    IMPORT_ERRORS.append(f"Fine-tuning module error: {e}")


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Gameplay UX Analyzer",
    page_icon="🎮",
    layout="wide"
)


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------

DATA_FOLDER = Path("data")
MODELS_FOLDER = Path("models")


def setup_data_folder() -> Path:
    DATA_FOLDER.mkdir(exist_ok=True)
    return DATA_FOLDER


def setup_models_folder() -> Path:
    MODELS_FOLDER.mkdir(exist_ok=True)
    return MODELS_FOLDER


def save_uploaded_file(uploaded, dest: Path) -> bool:
    try:
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def run_pipeline(data_folder: Path, game_context: str):
    """
    Full pipeline:
    - YOLO gameplay → gameplay_log.json
    - FER emotions → emotion_log.json
    - Merge logs → merged_log.json
    - LLM → UX report (string)
    """
    gp_video = data_folder / "gameplay_input.mp4"
    face_video = data_folder / "face_input.mp4"

    gameplay_log = data_folder / "gameplay_log.json"
    emotion_log = data_folder / "emotion_log.json"
    merged_path = data_folder / "merged_log.json"

    try:
        # 1) YOLO Gameplay
        with st.spinner("🎮 Running YOLO gameplay analysis..."):
            analyze_gameplay(str(gp_video), str(gameplay_log))

        # 2) FER Emotions
        with st.spinner("😊 Running facial emotion analysis..."):
            analyze_face(str(face_video), str(emotion_log))

        # 3) Merge Logs
        with st.spinner("🔗 Merging gameplay + emotion timelines..."):
            merged = merge_logs(
                str(gameplay_log),
                str(emotion_log),
                str(merged_path),
                max_time_diff=2.0,
            )

        # 4) LLM UX Report
        with st.spinner("🤖 Generating UX insights..."):
            report = generate_ux_report(
                merged_log_path=str(merged_path),
                game_context=game_context
            )

        # Save UX report to file for export tab
        ux_report_path = data_folder / "ux_report.txt"
        with open(ux_report_path, "w") as f:
            f.write(report)

        return merged, report

    except Exception as e:
        st.error("❌ Pipeline failed")
        st.code(traceback.format_exc())
        return None, None


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

def main():
    st.title("🎮 Gameplay UX Analyzer + YOLO Fine-Tuner")

    # Show any import issues, but don't hard-stop the app
    if IMPORT_ERRORS:
        st.warning("Some components failed to import:")
        for err in IMPORT_ERRORS:
            st.caption(f"• {err}")

    HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))

    # Top-level tabs (Option A)
    main_tab, finetune_tab, export_tab = st.tabs(
        ["🧪 Gameplay Analyzer", "🛠 YOLO Fine-Tuner", "☁️ Export / Downloads"]
    )

    # =====================================================
    # TAB 1 — GAMEPLAY ANALYZER
    # =====================================================
    with main_tab:
        st.header("🧪 Gameplay UX Analyzer")

        st.markdown("""
        Upload a **gameplay video** + **face video**, then run the pipeline to:
        - Detect gameplay events (enemies, combat spikes, deaths, stagnation)
        - Detect player emotions over time
        - Merge both into one timeline
        - Generate an AI-powered UX / QA report
        """)

        if not ANALYZER_OK:
            st.error("Core analyzer modules are not available. Check the error messages above.")
        if not HAS_OPENAI_KEY:
            st.error("Environment variable `OPENAI_API_KEY` is not set. The UX report step will fail until you set it.")

        st.divider()

        # Sidebar-like config inside this tab
        col_cfg, col_upload = st.columns([1, 2])

        with col_cfg:
            st.subheader("⚙️ Game Context")
            context = st.text_area(
                "Describe the scenario being tested:",
                "Boss fight demo in an action game. Player is new to this encounter.",
                height=120
            )

        with col_upload:
            st.subheader("📤 Upload Videos")

            col1, col2 = st.columns(2)
            with col1:
                gp_file = st.file_uploader("Gameplay Video", type=["mp4", "mov", "avi"])
            with col2:
                face_file = st.file_uploader("Face Video (player face)", type=["mp4", "mov", "avi"])

        ready = gp_file and face_file and ANALYZER_OK and HAS_OPENAI_KEY

        st.divider()

        if st.button("🚀 Run Full Analysis", disabled=not ready, use_container_width=True):
            data_folder = setup_data_folder()

            # Save uploaded videos
            ok1 = save_uploaded_file(gp_file, data_folder / "gameplay_input.mp4")
            ok2 = save_uploaded_file(face_file, data_folder / "face_input.mp4")

            if not (ok1 and ok2):
                st.error("Failed to save uploaded files. Fix this and retry.")
            else:
                merged, report = run_pipeline(data_folder, context)

                if merged and report:
                    st.success("🎉 Analysis complete!")
                    st.balloons()

                    # Keep in session for Export tab
                    st.session_state["merged_log"] = merged
                    st.session_state["ux_report"] = report

        # If we already have results, show them in sub-tabs
        if "merged_log" in st.session_state and "ux_report" in st.session_state:
            merged = st.session_state["merged_log"]
            report = st.session_state["ux_report"]

            st.divider()
            st.subheader("📊 Analysis Results")

            report_tab, timeline_tab, stats_tab, debug_tab = st.tabs(
                ["📄 UX Report", "📋 Timeline", "📈 Stats", "🎛 Visual Debug"]
            )

            # UX REPORT
            with report_tab:
                st.subheader("AI-Generated UX / QA Report")
                st.write(report)

            # TIMELINE
            with timeline_tab:
                st.subheader("Merged Gameplay + Emotion Timeline")

                max_show = st.slider(
                    "How many entries to preview?",
                    min_value=5,
                    max_value=min(200, len(merged)),
                    value=min(20, len(merged))
                )
                st.json(merged[:max_show])

                st.caption("Full merged_log.json is available in the Export / Downloads tab.")

            # STATS
            with stats_tab:
                st.subheader("Event + Emotion Statistics")

                event_counts = {}
                emotion_counts = {}

                for entry in merged:
                    event = entry.get("game_event", "unknown")
                    emo = entry.get("emotion_state", "unknown")
                    event_counts[event] = event_counts.get(event, 0) + 1
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

                col_e, col_m = st.columns(2)
                with col_e:
                    st.markdown("### 🎮 Game Events")
                    st.json(event_counts)
                with col_m:
                    st.markdown("### 🙂 Emotion States")
                    st.json(emotion_counts)

            # VISUAL DEBUG
            with debug_tab:
                st.subheader("🎛 Visual Debug Viewer")

                st.markdown("""
                This view helps you **debug correlations** between events and emotion states.
                """)

                # Simple filter controls
                unique_events = sorted({e.get("game_event", "unknown") for e in merged})
                unique_emotions = sorted({e.get("emotion_state", "unknown") for e in merged})

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    event_filter = st.multiselect(
                        "Filter by game event (optional):",
                        options=unique_events,
                        default=[]
                    )
                with col_f2:
                    emotion_filter = st.multiselect(
                        "Filter by emotion state (optional):",
                        options=unique_emotions,
                        default=[]
                    )

                filtered = []
                for e in merged:
                    if event_filter and e.get("game_event") not in event_filter:
                        continue
                    if emotion_filter and e.get("emotion_state") not in emotion_filter:
                        continue
                    filtered.append(e)

                st.markdown(f"Showing **{len(filtered)}** / {len(merged)} entries after filters")
                st.json(filtered[:50])

    # =====================================================
    # TAB 2 — YOLO FINE-TUNER
    # =====================================================
    with finetune_tab:
        st.header("🛠 YOLOv8 Fine-Tuner")

        st.markdown("""
        Upload a **YOLO-formatted dataset** of your game (images + labels)
        to fine-tune a custom detector (e.g., enemies, pickups, hazards, UI elements).

        This makes the gameplay analyzer **much more accurate** for a specific title.
        """)

        if not FINETUNER_OK:
            st.error("Fine-tuning module (`finetuner.py`) is not available. Check import errors above.")
        else:
            dataset_zip = st.file_uploader("Upload YOLO Dataset (.zip)", type=["zip"])

            colA, colB = st.columns(2)
            with colA:
                model_choice = st.selectbox(
                    "Base YOLOv8 model",
                    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
                )
                epochs = st.number_input(
                    "Training epochs",
                    min_value=1,
                    max_value=200,
                    value=10
                )
                imgsz = st.number_input(
                    "Image size (imgsz)",
                    min_value=320,
                    max_value=1280,
                    value=640,
                    step=32
                )

            with colB:
                st.info("""
                **Expected dataset structure (inside the .zip):**

                ```
                images/train
                images/val
                labels/train
                labels/val
                ```

                Labels must be standard YOLO txt files.
                """)

            if dataset_zip:
                DATA_ROOT = setup_data_folder() / "training"
                DATA_ROOT.mkdir(parents=True, exist_ok=True)

                zip_path = DATA_ROOT / "dataset.zip"
                with open(zip_path, "wb") as f:
                    f.write(dataset_zip.getbuffer())

                st.success("📦 Dataset uploaded. Ready to fine-tune.")

                if st.button("🚀 Start Fine-Tuning", use_container_width=True):
                    models_dir = setup_models_folder()
                    save_path = models_dir / "custom_yolov8.pt"

                    with st.spinner("📂 Extracting dataset..."):
                        extracted = extract_zip(str(zip_path), str(DATA_ROOT))

                    with st.spinner("📝 Generating dataset YAML..."):
                        yaml_path = create_dataset_yaml(extracted)

                    with st.spinner("🧠 Training YOLOv8 (this can take a while)..."):
                        train_yolov8(
                            base_model=model_choice,
                            data_yaml=str(yaml_path),
                            epochs=int(epochs),
                            imgsz=int(imgsz),
                            save_path=str(save_path)
                        )

                    st.success(f"🎉 Training complete! Model saved to {save_path}")

                    # Store flag so Export tab can offer model download
                    st.session_state["custom_model_path"] = str(save_path)

                    with open(save_path, "rb") as f:
                        st.download_button(
                            "📥 Download Custom YOLOv8 Model",
                            data=f.read(),
                            file_name="custom_yolov8.pt"
                        )

    # =====================================================
    # TAB 3 — EXPORT / DOWNLOADS
    # =====================================================
    with export_tab:
        st.header("☁️ Export / Downloads")

        st.markdown("""
        Download the outputs of your analysis and training:
        - UX report
        - Merged gameplay + emotion timeline
        - Raw gameplay & emotion logs
        - Fine-tuned YOLO model (if trained)
        """)

        data_folder = DATA_FOLDER
        models_folder = MODELS_FOLDER

        # UX REPORT
        ux_report_text = st.session_state.get("ux_report")
        ux_report_file = data_folder / "ux_report.txt"

        st.subheader("📄 UX Report")
        if ux_report_text:
            st.download_button(
                "Download UX Report (txt)",
                data=ux_report_text,
                file_name="ux_report.txt"
            )
        elif ux_report_file.exists():
            with open(ux_report_file, "r") as f:
                txt = f.read()
            st.download_button(
                "Download UX Report (txt)",
                data=txt,
                file_name="ux_report.txt"
            )
        else:
            st.caption("No UX report found yet. Run the Gameplay Analyzer first.")

        st.divider()

        # MERGED LOG
        merged_file = data_folder / "merged_log.json"
        st.subheader("📋 Merged Timeline JSON")
        if merged_file.exists():
            with open(merged_file, "r") as f:
                merged_raw = f.read()
            st.download_button(
                "Download merged_log.json",
                data=merged_raw,
                file_name="merged_log.json"
            )
        else:
            st.caption("No merged_log.json yet.")

        # RAW LOGS
        st.subheader("📂 Raw Logs")
        gameplay_log = data_folder / "gameplay_log.json"
        emotion_log = data_folder / "emotion_log.json"

        col_g, col_e = st.columns(2)
        with col_g:
            if gameplay_log.exists():
                with open(gameplay_log, "r") as f:
                    g_raw = f.read()
                st.download_button(
                    "Download gameplay_log.json",
                    data=g_raw,
                    file_name="gameplay_log.json"
                )
            else:
                st.caption("No gameplay_log.json yet.")
        with col_e:
            if emotion_log.exists():
                with open(emotion_log, "r") as f:
                    e_raw = f.read()
                st.download_button(
                    "Download emotion_log.json",
                    data=e_raw,
                    file_name="emotion_log.json"
                )
            else:
                st.caption("No emotion_log.json yet.")

        st.divider()

        # CUSTOM MODEL
        st.subheader("🧠 Fine-Tuned YOLO Model")
        custom_model_path = st.session_state.get("custom_model_path")
        if custom_model_path and Path(custom_model_path).exists():
            with open(custom_model_path, "rb") as f:
                st.download_button(
                    "Download custom_yolov8.pt",
                    data=f.read(),
                    file_name="custom_yolov8.pt"
                )
        else:
            # Check on disk just in case
            candidate = models_folder / "custom_yolov8.pt"
            if candidate.exists():
                with open(candidate, "rb") as f:
                    st.download_button(
                        "Download custom_yolov8.pt",
                        data=f.read(),
                        file_name="custom_yolov8.pt"
                    )
            else:
                st.caption("No fine-tuned model found yet. Train one in the YOLO Fine-Tuner tab.")


if __name__ == "__main__":
    main()
