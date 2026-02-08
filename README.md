#DOMAIN UPDATE
# Clash Royale- Gameplay UX & Emotion QA Analyzer
### Hackathon Edition — Clash Royale Domain Extension
<img width="1422" height="921" alt="Screenshot 2026-02-08 at 4 00 22 AM" src="https://github.com/user-attachments/assets/4420907a-8f72-44af-aa4f-1942a08086b6" />

This project is a **multimodal gameplay UX & QA analysis system** that combines:

- 🎥 Gameplay video understanding (computer vision)
- 🙂 Player facial emotion recognition
- 🤖 AI-generated UX / QA insights

For this hackathon, the system was extended with a **Clash Royale–specific domain module**, demonstrating how a generic video-based analyzer can be adapted to a real commercial game using **custom-trained models and game-aware logic**.

---

## 🆕 Clash Royale Domain Module (Hackathon Update)

### What is this?

The **Clash Royale Domain Module** is a domain-aware extension that enables the system to understand **actual Clash Royale gameplay semantics**, rather than generic objects or motion.

Instead of detecting abstract entities, the analyzer now detects and reasons about:

- **Specific troops** (e.g. *Mega Knight, Hog Rider, Musketeer, Skeletons*)
- **Spells** (e.g. *Rage, Poison*)
- **Buildings & towers**
- **UI-only and decision-making states**
- **Gameplay pacing signals** (troop spikes, spell bursts, stagnation)

Each detected gameplay event is **time-aligned with player emotion**, allowing the system to explain *how gameplay decisions correlate with emotional response*.

> In short:  
> **This turns raw gameplay video into structured, card-level UX telemetry — without engine access.**

<img width="1550" height="528" alt="Screenshot 2026-02-08 at 4 53 24 AM" src="https://github.com/user-attachments/assets/e254c92a-bd87-4f98-8561-1defc0abdd4b" />
<img width="1505" height="179" alt="Screenshot 2026-02-08 at 4 53 10 AM" src="https://github.com/user-attachments/assets/78dc47c7-95e3-48cb-9c65-4cc5927cf541" />

---

## Why This Matters 

Most game analytics rely on:
- Engine telemetry (often unavailable), or
- Generic computer vision that lacks gameplay meaning

This project demonstrates a third approach:
-  **Video-only analysis**
-  **Domain-specific perception**
- **Emotion-aware UX interpretation**

Using Clash Royale as a case study, the system can answer questions like:
- *Which cards or spells trigger frustration or confidence?*
- *When do players hesitate or disengage?*
- *Which moments correlate with emotional spikes during combat?*

The same architecture can be extended to **other games** by training new models and defining new domain profiles.

---

##  Model & System Changes (What Was Added)

### 1. Custom YOLOv8 Model (Clash Royale)

- Trained a **custom YOLOv8 detector** on a Clash Royale dataset
- Model detects **troops, spells, towers, buildings, and UI elements**
- Replaces generic COCO classes with **game-specific labels**

### 2. Domain-Aware Gameplay Logic

The original gameplay analyzer was refactored to:
- Log **specific troop and spell names per frame**
- Detect **troop spikes**, **spell casts**, and **UI-only states**
- Preserve the original generic (COCO) pipeline as a fallback

Clash Royale is implemented as a **domain profile**, not a hard-coded fork — keeping the system modular and extensible.

### 3. Emotion-Aligned Timeline

- Facial emotion recognition runs independently
- Gameplay and emotion logs are merged by timestamp
- The final timeline links **what happened in-game** with **how the player felt**

---

## 📊 Example Outputs

**Structured gameplay events**
- Troop deployments (by card name)
- Spell usage bursts
- Long hesitation / stagnation periods

**Emotion-aware UX insights**
- Confidence spikes during spell combos
- Frustration during prolonged UI-only states
- Emotional response to specific card usage

---

##  Dataset Credit

This Clash Royale detector was trained using the following open dataset:

**Clash Royale Detection Dataset**  
📎 https://github.com/wty-yy/Clash-Royale-Detection-Dataset  

All credit for dataset creation and labeling goes to the original authors.

---


---

##  Future Work

- Deeper **temporal reasoning** across multi-second decision windows
- Player modeling across matches
- Generalizing the domain module framework to additional games
- Research into **emotion-conditioned gameplay difficulty and pacing**

---

*This hackathon extension demonstrates how video-only, domain-aware AI can enable scalable UX research for games without engine access.*


<img width="1417" height="916" alt="Screenshot 2026-02-08 at 4 01 04 AM" src="https://github.com/user-attachments/assets/7a149f24-ca32-4df2-b986-86823b1b48af" />


___________________________________________________________________________________________




FIRST PROTOTYPE BELOW



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
