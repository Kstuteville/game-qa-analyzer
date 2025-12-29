# Emotion-Trace-QA
Game QA Analyzer
Emotion-Aware Gameplay Analysis for Scalable Game QA
Computer-vision–driven QA system that analyzes gameplay footage and correlates in-game events with player emotional states, producing structured, timestamped insights without manual video review.


🚀 Overview
Game QA Analyzer automates playtest analysis by combining:

- Computer Vision (what happens on screen)
- Emotion Inference (how players react)
- Temporal Alignment (when and why)

Designed for game studios, QA teams, and researchers working with pre-release or private builds where traditional analytics fall short.


## 🎥 Demo

[![Emotion Trace QA Demo](https://img.youtube.com/vi/JMtA4VYmuvw/maxresdefault.jpg)](https://www.youtube.com/watch?v=JMtA4VYmuvw)



✨ Features

✔ Gameplay Event Detection — YOLOv8-powered object & event recognition
✔ Emotion Inference — Facial expression analysis (FER) with planned audio/physiological expansion
✔ Timestamped QA Logs — Structured event–emotion correlations
✔ Custom Fine-Tuning — Game-specific models without ML expertise
✔ Privacy-First — Runs entirely on-device (NVIDIA GB10)


🧠 How It Works
1. Gameplay Event Detection

Uses YOLOv8 to identify:
- Enemy encounters
- UI states
- Player deaths/failures
- Items/objectives
- Combat intensity

2. Emotion Inference
Analyzes facecam footage to detect:
- Calm
- Focused
- Frustrated
- Stressed
- Confused

4. Temporal Correlation
Links events to emotional states:
json{
  "time": 42.31,
  "game_event": "player_death", 
  "emotion_state": "frustrated",
  "confidence": 0.81
} 

📊 Output

Timestamped JSON logs for QA dashboards
Event–emotion correlations for UX analysis
Accessibility insights for inclusive design
Emotional pacing data for difficulty tuning


🔒 Privacy & Security
All training and inference runs locally on NVIDIA GB10:

No gameplay footage uploaded to third parties
Pre-release builds remain secure
NDA-compliant workflow
Studio-controlled data pipeline


🛠 Tech Stack

Python
YOLOv8 (Ultralytics)
OpenCV
FER (Facial Emotion Recognition)
PyTorch
NVIDIA GPU acceleration (GB10)

Planned: NVIDIA VLMs, multimodal fusion (audio + vision), interactive QA debugger

🧪 Fine-Tuning Workflow
Studios can fine-tune models without ML expertise:
- Record gameplay clips
- Label game-specific events
- Fine-tune YOLOv8 locally on GB10
- Run QA analysis offline

No external APIs. No cloud training. No data leakage.

🎯 Use Cases
- Faster QA iteration
- Emotion-aware difficulty adjustment (DDA)
- Accessibility testing
- Player frustration modeling
- UX research for unreleased titles


📌 Project Status

- ✅ Core CV pipeline implemented
- ✅ YOLOv8 fine-tuning operational
- ✅ Structured QA logs generated
- 🚧 Visual QA debugger (in progress)
- 🚧 Multimodal emotion fusion (planned)


👩‍💻 Author
Kaylie Stuteville
MS Integrated Design & Media — NYU Tandon
Focus: AI-driven game systems, emotion-aware NPCs, scalable QA tools

🔗 Why This Matters
Studios log what players do — but not how they feel.
Game QA Analyzer bridges quantitative telemetry with qualitative player experience, enabling emotion-driven game design at scale.
