## game-qa-analyzer — Copilot instructions

Purpose
- Help an AI coding agent get productive quickly in this repo: a Streamlit-based Gameplay UX analyzer + optional YOLO fine-tuner.

Quick start (dev)
- Install dependencies: `pip install -r requirements.txt` (project contains notes about CPU/GPU `torch` installs).
- Ensure `OPENAI_API_KEY` is set in environment (the app reads OpenAI via env var).
- Run the UI: `streamlit run src/app.py`

Big picture / architecture
- Entry UI: `src/app.py` (Streamlit). It wires a 4-step pipeline in `run_pipeline()`:
  1. `yolo_analyzer.analyze_gameplay()` → writes `gameplay_log.json`
  2. `fer_analyzer.analyze_face()` → writes `emotion_log.json`
  3. `merge_logs.merge_logs()` → writes `merged_log.json`
  4. `llm_agent.generate_ux_report()` → returns a report string saved as `ux_report.txt`
- Optional fine-tuning: `src/finetuner.py` contains helpers (`extract_zip`, `create_dataset_yaml`, `train_yolov8`) used by the Fine-Tuner tab.

Key files and responsibilities
- `src/app.py` — Streamlit UI, tabs, upload flow, orchestration (shows import errors but doesn't hard-fail when modules are unavailable).
- `src/yolo_analyzer.py` — CPU/GPU-safe YOLOv8 usage (loads `yolov8n.pt` by default). Produces an array of event dicts and writes `gameplay_log.json`.
- `src/fer_analyzer.py` — FER-based emotion extraction; produces per-frame emotion entries and writes `emotion_log.json`.
- `src/merge_logs.py` — Aligns gameplay events to the nearest emotion entry and writes `merged_log.json`. Output entries look like:
  `{ "time": 12.34, "game_event": "enemy", "emotion_state": "angry", "confidence": 0.87 }`
- `src/llm_agent.py` — Calls OpenAI Responses API (client = `OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`) and uses `gpt-4.1-mini` in current code.
- `src/finetuner.py` — ZIP extraction + YAML generation + wrapper to `ultralytics.YOLO.train()`; expects YOLO-format labels.

Data formats / conventions
- All analyzer outputs are JSON arrays of dicts. Keep functions returning lists and also writing JSON to disk (this pattern is used across analyzers).
- Common log filenames (used across UI): `gameplay_log.json`, `emotion_log.json`, `merged_log.json`, `ux_report.txt`, `custom_yolov8.pt` (models/).
- YOLO dataset zip expected structure (inside .zip): `images/train`, `images/val`, `labels/train`, `labels/val`. `create_dataset_yaml()` expects the extracted folder to contain a `train` folder; it then writes `data.yaml` next to that folder.

Integration points & external deps
- Ultralytics YOLO (`ultralytics`) — used in `yolo_analyzer.py` and `finetuner.py`. Local `yolov8n.pt` is included; ultralytics will auto-download if missing.
- FER (`fer`, `facenet-pytorch`) — used by `fer_analyzer.py` for emotion detection.
- OpenAI Python client — `src/llm_agent.py` uses `OpenAI(...).responses.create(...)`. Requires `OPENAI_API_KEY` env var.
- CV/video stack: `opencv-python`, `ffmpeg-python` (notes in requirements), `numpy`, `Pillow`.

Developer workflows / debugging tips
- To reproduce the full pipeline locally: place `gameplay_input.mp4` and `face_input.mp4` in `data/` (or upload via UI) then `streamlit run src/app.py`.
- If a submodule fails to import at startup, `src/app.py` collects messages in `IMPORT_ERRORS` and shows them in the UI; check those messages first.
- For quick CLI runs:
  - `python src/yolo_analyzer.py path/to/video.mp4`
  - `python src/fer_analyzer.py path/to/face_video.mp4`
  - `python src/merge_logs.py gameplay_log.json emotion_log.json merged_log.json`
- Fine-tuning notes:
  - Upload a zip with the YOLO-format dataset via the Fine-Tuner tab.
  - `finetuner.create_dataset_yaml()` writes `data.yaml` at the dataset root. `train_yolov8()` uses `ultralytics.YOLO.train()` and then moves the best checkpoint to `models/custom_yolov8.pt`.
  - Training can be long and may require a GPU and proper `torch` wheel. See `requirements.txt` for hints and the included notes about `torch` install.

Conventions for changes and tests
- Keep log outputs as lists of dicts (backwards compatibility with the LLM prompt in `llm_agent.py`).
- When changing keys in logs, update `merge_logs.py` (it already contains a few corrective lookups for key names).
- Add unit tests around `merge_logs.load_json()` and the nearest-neighbor merge logic (happy path + missing emotion entries) because downstream LLM prompts assume the merged format.

Other notes
- `requirements.txt` contains extra developer comments (GPU instructions, example pip commands). Follow those hints for platform-specific installs.
- Avoid putting API keys in repository files — the app reads `OPENAI_API_KEY` from env vars. There is a `key.txt` in the repo root; do NOT commit secrets — prefer env vars or `.env` loaded via `python-dotenv`.

If anything in these notes is unclear or you want more details (examples of merged JSON entries, unit tests added, CI steps), tell me which section to expand and I'll iterate.
