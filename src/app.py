"""
Gameplay UX Analyzer + YOLO Fine-Tuner
Safe Mac + Dell version
"""

import streamlit as st
from pathlib import Path
import traceback
import json
import os
from datetime import datetime
from collections import Counter  # (optional)

ANALYZER_OK = True
IMPORT_ERRORS = []
RUNS_FOLDER = Path("runs")
RUNS_FOLDER.mkdir(exist_ok=True)
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
st.set_page_config(
   page_title="Gameplay UX Analyzer",
   page_icon="🎮",
   layout="wide"
)
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

def find_closest_log_entry(merged_log, target_time):
    """
    Find the merged log entry closest to the target timestamp.
    Returns the entry and its index.
    """
    if not merged_log:
        return None, -1
    closest_entry = None
    closest_idx = -1
    min_diff = float('inf')
    for idx, entry in enumerate(merged_log):
        timestamp = entry.get('time', 0)
        diff = abs(timestamp - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_entry = entry
            closest_idx = idx
    return closest_entry, closest_idx


def get_video_duration_estimate(merged_log):
    """
    Estimate video duration from merged log timestamps.
    Returns max timestamp found, or 60.0 as fallback.
    """
    if not merged_log:
        return 60.0
    max_time = max(entry.get('time', 0) for entry in merged_log)
    return max(max_time, 60.0)


def run_pipeline(data_folder: Path, game_context: str, profile: dict):
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
       with st.spinner("Running YOLO gameplay analysis..."):
           analyze_gameplay(str(gp_video), str(gameplay_log), profile=profile)
       # 2) FER Emotions
       with st.spinner("Running facial emotion analysis..."):
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
       st.error("Pipeline failed")
       st.code(traceback.format_exc())
       return None, None


def list_runs() -> list[str]:
    """Return run folder names newest-first."""
    if not RUNS_FOLDER.exists():
        return []
    valid_runs = []
    for p in RUNS_FOLDER.iterdir():
        if not p.is_dir():
            continue
        if (p / "merged_log.json").exists() and (p / "ux_report.txt").exists():
            valid_runs.append(p.name)
    valid_runs.sort(reverse=True)
    return valid_runs


def create_new_run_folder() -> tuple[str, Path]:
    """Create a new unique run folder and return (run_id, path)."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = RUNS_FOLDER / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_id, run_path

def load_run_into_session(run_id: str) -> bool:
    """Load merged_log + report from a run folder into session state."""
    run_path = RUNS_FOLDER / run_id
    merged_path = run_path / "merged_log.json"
    report_path = run_path / "ux_report.txt"
    if not (merged_path.exists() and report_path.exists()):
        return False

    with open(merged_path, "r") as f:
        st.session_state["merged_log"] = json.load(f)
    with open(report_path, "r") as f:
        st.session_state["ux_report"] = f.read()

    st.session_state["current_run"] = run_id
    st.session_state["play_time"] = 0.0
    return True

def get_active_run_path() -> Path:
    """Return the currently selected run folder path, or empty Path if none."""
    run_id = st.session_state.get("current_run", "")
    if not run_id:
        return Path("")
    return RUNS_FOLDER / run_id

def main():
   st.title(" Gameplay UX Analyzer + YOLO Fine-Tuner")
   # Show any import issues, but don't hard-stop the app
   if IMPORT_ERRORS:
       st.warning("Some components failed to import:")
       for err in IMPORT_ERRORS:
           st.caption(f"• {err}")
   HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
   main_tab, finetune_tab, export_tab = st.tabs(
       ["Gameplay Analyzer", " YOLO Fine-Tuner", "☁️ Export / Downloads"]
   )
   with main_tab:
       st.header(" Gameplay UX Analyzer")
       st.subheader(" Analysis History")
       runs = list_runs()
       selected_run = st.selectbox("Load a previous run:", [""] + runs)
       col_hist_a, col_hist_b = st.columns([1, 2])
       with col_hist_a:
                if st.button("🔄 Refresh run list"):
                    st.rerun()
       with col_hist_b:
            if selected_run:
                ok = load_run_into_session(selected_run)
                if ok:
                    st.success(f"Loaded run: {selected_run}")
                else:
                    st.error("That run is missing merged_log.json or ux_report.txt.")

       active_run = st.session_state.get("current_run", "")
       if active_run:
          st.caption(f" Active run: {active_run}")

       from profiles import GENERIC_COCO, CLASH_ROYALE
       profile_label = st.selectbox("Game Analysis Profile",
                                    options=[GENERIC_COCO["label"], CLASH_ROYALE["label"]],
           index=0
       
       )
       PROFILE = GENERIC_COCO if profile_label == GENERIC_COCO["label"] else CLASH_ROYALE
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
           st.subheader(" Game Context")
           context = st.text_area(
               "Describe the scenario being tested:",
               "Boss fight demo in an action game. Player is new to this encounter.",
               height=120
           )

       with col_upload:
           st.subheader(" Upload Videos")
           col1, col2 = st.columns(2)
           with col1:
               gp_file = st.file_uploader("Gameplay Video", type=["mp4", "mov", "avi"])
           with col2:
               face_file = st.file_uploader("Face Video (player face)", type=["mp4", "mov", "avi"])
       ready = gp_file and face_file and ANALYZER_OK and HAS_OPENAI_KEY
       st.divider()
       if st.button(" Run Full Analysis", disabled=not ready, use_container_width=True):
           run_id, data_folder = create_new_run_folder()
           st.session_state["current_run"] = run_id
           # Save uploaded videos
           ok1 = save_uploaded_file(gp_file, data_folder / "gameplay_input.mp4")
           ok2 = save_uploaded_file(face_file, data_folder / "face_input.mp4")
           if not (ok1 and ok2):
               st.error("Failed to save uploaded files. Fix this and retry.")
           else:
               merged, report = run_pipeline(data_folder, context, PROFILE)
               if merged and report:
                   st.success(" Analysis complete!")
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
               ["UX Report", "Timeline", "Stats", "Visual Debug"]
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
          # STATS
           with stats_tab:
                from collections import Counter

                st.subheader("Event + Emotion Statistics")
                event_counts = Counter()
                emotion_counts = Counter()
                # Clash-specific rollups
                troop_counts = Counter()
                spell_counts = Counter()
                building_counts = Counter()
                # Helpful: how often a troop appears alongside each emotion
                troop_by_emotion = {}  # dict[str, Counter]
                for entry in merged:
                    # --- base counts ---
                    event = entry.get("game_event", "unknown")
                    emo = entry.get("emotion_state", "unknown")
                    event_counts[event] += 1
                    emotion_counts[emo] += 1
                    # --- details (new merge_logs should keep this) ---
                    d = entry.get("game_details", {}) or {}
                    troops = d.get("troops", []) or []
                    spells = d.get("spells", []) or []
                    buildings = d.get("buildings", []) or []
                    for t in troops:
                        troop_counts[t] += 1
                        if t not in troop_by_emotion:
                            troop_by_emotion[t] = Counter()
                        troop_by_emotion[t][emo] += 1
                    for s in spells:
                        spell_counts[s] += 1
                    for b in buildings:
                        building_counts[b] += 1
                # --- show base counts ---
                col_e, col_m = st.columns(2)
                with col_e:
                    st.markdown("###  Game Events")
                    st.json(dict(event_counts))
                with col_m:
                    st.markdown("###  Emotion States")
                    st.json(dict(emotion_counts))

                st.divider()

                # --- show Clash rollups (even if empty, it will be obvious) ---
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("### Top Troops")
                    st.json(dict(troop_counts.most_common(25)))
                with col_b:
                    st.markdown("###  Top Spells")
                    st.json(dict(spell_counts.most_common(25)))
                with col_c:
                    st.markdown("### Top Buildings")
                    st.json(dict(building_counts.most_common(25)))

                st.divider()
                # --- optional: troop ↔ emotion correlation quick view ---
                st.markdown("### 🔁 Troop ↔ Emotion Breakdown (top 12 troops)")
                top_troops = [t for t, _ in troop_counts.most_common(12)]
                breakdown = {}
                for t in top_troops:
                    breakdown[t] = dict(troop_by_emotion.get(t, Counter()))
                st.json(breakdown)

           # VISUAL DEBUG
           with debug_tab:
               st.subheader("🎛 Visual Debug Viewer")

               st.markdown("""
               This view helps you **debug correlations** between events and emotion states.
               Use the timestamp scrubber to navigate through the videos and see synchronized data.
               """)
               # NEW: Visual Timestamp Debugger
               st.divider()
               st.markdown("### Timestamp Scrubber")
               # Check if videos exist
               data_folder = get_active_run_path()
               gameplay_video_path = data_folder / "gameplay_input.mp4"
               face_video_path = data_folder / "face_input.mp4"
               videos_available = gameplay_video_path.exists() and face_video_path.exists()      
               if videos_available:
                   # Estimate video duration from merged log
                   video_duration = get_video_duration_estimate(merged)
                   # Timestamp slider
                   selected_time = st.slider(
                       "Select timestamp (seconds):",
                       min_value=0.0,
                       max_value=float(video_duration),
                       value=0.0,
                       step=0.5,
                       format="%.1f s"
                   )
                   # Find closest log entry
                   closest_entry, entry_idx = find_closest_log_entry(merged, selected_time)
                   # Display videos and log data
                   col_vid1, col_vid2, col_data = st.columns([2, 2, 1])
                   with col_vid1:
                       st.markdown("#### Gameplay")
                       st.video(str(gameplay_video_path), start_time=int(selected_time))
                   with col_vid2:
                       st.markdown("####  Player Face")
                       st.video(str(face_video_path), start_time=int(selected_time))
                   with col_data:
                        st.markdown("#### Data at Timestamp")
                        if closest_entry:
                            st.metric("Timestamp", f"{closest_entry.get('time', 0):.2f}s")
                            st.metric("Game Event", closest_entry.get('game_event', 'N/A'))
                            st.metric("Emotion", closest_entry.get('emotion_state', 'N/A'))
                            confidence = closest_entry.get('confidence')
                            if confidence is not None:
                                st.metric("Confidence", f"{confidence:.2%}")
                            #  NEW: show detected entities
                            d = closest_entry.get("game_details", {}) or {}
                            troops = d.get("troops", []) or []
                            spells = d.get("spells", []) or []
                            buildings = d.get("buildings", []) or []
                            if troops:
                                st.caption("Troops: " + ", ".join(troops[:12]) + ("..." if len(troops) > 12 else ""))
                            if spells:
                                st.caption("Spells: " + ", ".join(spells[:12]) + ("..." if len(spells) > 12 else ""))
                            if buildings:
                                st.caption("Buildings: " + ", ".join(buildings[:12]) + ("..." if len(buildings) > 12 else ""))

                            st.caption(f"Entry #{entry_idx + 1} of {len(merged)}")
                        else:
                            st.info("No log data available")
                                    
                   # Show detailed entry data below
                   st.divider()
                   if closest_entry:
                       with st.expander("🔍 View Full Log Entry Details"):
                           st.json(closest_entry)
               else:
                   st.warning("Videos not found. Please run the analysis first to generate gameplay_input.mp4 and face_input.mp4")
               # Original filter functionality
               st.divider()
               st.markdown("###Filter Timeline Data")
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
    
   with finetune_tab:
       st.header(" YOLOv8 Fine-Tuner")
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
               st.success(" Dataset uploaded. Ready to fine-tune.")
               if st.button(" Start Fine-Tuning", use_container_width=True):
                   models_dir = setup_models_folder()
                   save_path = models_dir / "custom_yolov8.pt"
                   with st.spinner(" Extracting dataset..."):
                       extracted = extract_zip(str(zip_path), str(DATA_ROOT))
                   with st.spinner(" Generating dataset YAML..."):
                       yaml_path = create_dataset_yaml(extracted)
                   with st.spinner(" Training YOLOv8 (this can take a while)..."):
                       train_yolov8(
                           base_model=model_choice,
                           data_yaml=str(yaml_path),
                           epochs=int(epochs),
                           imgsz=int(imgsz),
                           save_path=str(save_path)
                       )
                   st.success(f" Training complete! Model saved to {save_path}")
                   # Store flag so Export tab can offer model download
                   st.session_state["custom_model_path"] = str(save_path)
                   with open(save_path, "rb") as f:
                       st.download_button(
                           " Download Custom YOLOv8 Model",
                           data=f.read(),
                           file_name="custom_yolov8.pt"
                       )
   with export_tab:
       st.header(" Export / Downloads")
       st.markdown("""
       Download the outputs of your analysis and training:
       - UX report
       - Merged gameplay + emotion timeline
       - Raw gameplay & emotion logs
       - Fine-tuned YOLO model (if trained)
       """)
       active_run = st.session_state.get("current_run", "")
       if not active_run:      
            st.warning("No run selected yet. Load a previous run or run an analysis first.")
            st.stop()
       data_folder = get_active_run_path()
       models_folder = MODELS_FOLDER
       # UX REPORT
       ux_report_text = st.session_state.get("ux_report")
       ux_report_file = data_folder / "ux_report.txt"
       st.subheader(" UX Report")
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
       st.subheader(" Merged Timeline JSON")
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
       st.subheader("Raw Logs")
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
       st.subheader(" Fine-Tuned YOLO Model")
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
