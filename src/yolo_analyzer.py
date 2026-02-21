"""
yolo_analyzer.py — Clash Royale + COCO (profile-aware)

For Clash Royale:
- logs specific troop/building/spell names per sampled frame
- logs spikes and stagnation
For COCO:
- keeps  original heuristic events
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def load_model(profile: dict | None):
    if profile and profile.get("mode") == "clash":
        custom = Path("models") / "custom_yolov8.pt"
        if custom.exists():
            print(f"Loading Clash Royale custom model: {custom}")
            return YOLO(str(custom))
        else:
            print("⚠️ Clash profile selected but custom model not found, falling back to YOLOv8n.pt")

    print("Loading default YOLOv8n pretrained model...")
    return YOLO("yolov8n.pt")


def analyze_gameplay(
    video_path: str,
    output_json_path: str = "gameplay_log.json",
    profile: dict | None = None
) -> list:

    events: list[dict] = []

    # ---- Profile defaults ----
    CONF = float(profile.get("conf_threshold", 0.5)) if profile else 0.5
    FRAME_SAMPLE_RATE = int(profile.get("frame_sample_rate", 3)) if profile else 3
    MODE = profile.get("mode", "coco") if profile else "coco"
    CLASSES_OF_INTEREST = set(profile.get("classes_of_interest", [])) if profile else None
    SPELL_NAMES = set(profile.get("spell_names", [])) if profile else set()

    model = load_model(profile)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    # --- heuristics ---
    DARKNESS_THRESHOLD = 30
    STAGNATION_THRESHOLD = 0.02
    STAGNATION_DURATION = 2.0

    # COCO-style heuristics (kept) 
    ENEMY_CLASSES = {'person', 'dog', 'cat', 'horse', 'bear', 'zebra', 'giraffe', 'elephant', 'bird'}
    PROJECTILE_CLASSES = {'sports ball', 'frisbee', 'kite', 'baseball bat', 'baseball glove'}
    WEAPON_CLASSES = {'knife', 'baseball bat', 'tennis racket', 'sports ball'}

    # Clash categories
    UI_CLASSES = {
        "bar", "bar-level", "clock", "text", "emote", "selected",
        "elixir", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar",
        "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol"
    }
    TOWER_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}

    # not perfect, but good demo buckets (edit anytime)
    BUILDING_CLASSES = {
        "cannon", "tesla", "inferno-tower", "bomb-tower", "tombstone",
        "goblin-hut", "barbarian-hut", "elixir-collector", "x-bow", "mortar",
        "goblin-cage"
    }

    # For spike thresholds (demo-friendly)
    TROOP_SPIKE_THRESHOLD = int(profile.get("troop_spike_threshold", 6)) if profile else 6
    BUILDING_SPIKE_THRESHOLD = int(profile.get("building_spike_threshold", 3)) if profile else 3

    frame_idx = 0
    prev_frame_gray = None
    stagnation_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        if frame_idx % FRAME_SAMPLE_RATE == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(frame_gray))

            try:
                results = model.predict(frame, conf=CONF, imgsz=640, verbose=False)
            except Exception as e:
                print(f"YOLO error on frame {frame_idx}: {e}")
                frame_idx += 1
                continue


            detected = []  # list of (name, conf)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    class_name = model.names.get(cls_id, "unknown")
                    confidence = float(box.conf.item())

                    if confidence < CONF:
                        continue

                    # Optional filter for Clash: only keep labels you expect
                    if MODE == "clash" and CLASSES_OF_INTEREST:
                        if class_name not in CLASSES_OF_INTEREST:
                            continue

                    detected.append((class_name, confidence))

            # summarize per-class counts
            counts: dict[str, int] = {}
            for name, _c in detected:
                counts[name] = counts.get(name, 0) + 1

            if MODE == "clash":
                spells_seen = sorted({n for n in counts.keys() if n in SPELL_NAMES})
                towers_seen = sorted({n for n in counts.keys() if n in TOWER_CLASSES})
                ui_seen = sorted({n for n in counts.keys() if n in UI_CLASSES})
                buildings_seen = sorted({n for n in counts.keys() if n in BUILDING_CLASSES})

                # troops = everything else that isn’t tower/ui/spell/building
                troops_seen = sorted({
                    n for n in counts.keys()
                    if (n not in SPELL_NAMES)
                    and (n not in TOWER_CLASSES)
                    and (n not in UI_CLASSES)
                    and (n not in BUILDING_CLASSES)
                })

                troop_count = sum(counts.get(n, 0) for n in troops_seen)
                building_count = sum(counts.get(n, 0) for n in buildings_seen)

                # 1) Main per-frame “state” event: names + counts
                events.append({
                    "time": round(timestamp, 2),
                    "event": "clash_frame",
                    "details": {
                        "troops": troops_seen,
                        "troop_count": troop_count,
                        "buildings": buildings_seen,
                        "building_count": building_count,
                        "spells": spells_seen,
                        "towers": towers_seen,
                        "ui": ui_seen,
                        # useful for debugging: top detections
                        "top_detected": sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12],
                    }
                })

                # 2) Spell cast (kept as separate event for easy filtering)
                if spells_seen:
                    events.append({
                        "time": round(timestamp, 2),
                        "event": "spell_cast",
                        "details": {"spells": spells_seen}
                    })

                # 3) Troop / building spikes
                if troop_count >= TROOP_SPIKE_THRESHOLD:
                    events.append({
                        "time": round(timestamp, 2),
                        "event": "troop_spike",
                        "details": {"troop_count": troop_count, "troops": troops_seen}
                    })

                if building_count >= BUILDING_SPIKE_THRESHOLD:
                    events.append({
                        "time": round(timestamp, 2),
                        "event": "building_spike",
                        "details": {"building_count": building_count, "buildings": buildings_seen}
                    })

                # 4) “UI only” moments (no troops/buildings/spells)
                if troop_count == 0 and building_count == 0 and not spells_seen:
                    if ui_seen or towers_seen:
                        events.append({
                            "time": round(timestamp, 2),
                            "event": "ui_only",
                            "details": {"ui": ui_seen, "towers": towers_seen}
                        })

            else:
                enemy_count = sum(counts.get(n, 0) for n in ENEMY_CLASSES)
                projectile_count = sum(counts.get(n, 0) for n in PROJECTILE_CLASSES)
                weapon_count = sum(counts.get(n, 0) for n in WEAPON_CLASSES)

                if enemy_count > 0:
                    events.append({"time": round(timestamp, 2), "event": "enemy", "details": {"count": enemy_count}})
                if projectile_count > 0:
                    events.append({"time": round(timestamp, 2), "event": "projectile", "details": {"count": projectile_count}})
                if weapon_count > 0:
                    events.append({"time": round(timestamp, 2), "event": "weapon_like", "details": {"count": weapon_count}})
                if enemy_count >= 3:
                    events.append({"time": round(timestamp, 2), "event": "combat_spike", "details": {"count": enemy_count}})

                if not counts and mean_brightness < DARKNESS_THRESHOLD:
                    events.append({
                        "time": round(timestamp, 2),
                        "event": "death_or_loading_screen",
                        "details": {"brightness": round(mean_brightness, 2)}
                    })

            if prev_frame_gray is not None:
                diff = cv2.absdiff(frame_gray, prev_frame_gray)
                diff_norm = float(np.mean(diff)) / 255.0

                if diff_norm < STAGNATION_THRESHOLD:
                    if stagnation_start_time is None:
                        stagnation_start_time = timestamp
                    elif timestamp - stagnation_start_time >= STAGNATION_DURATION:
                        events.append({
                            "time": round(timestamp, 2),
                            "event": "stagnation_or_pause",
                            "details": {"duration": round(timestamp - stagnation_start_time, 2)}
                        })
                else:
                    stagnation_start_time = None

            prev_frame_gray = frame_gray.copy()

        frame_idx += 1

    cap.release()

    with open(output_json_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nSaved gameplay analysis to {output_json_path}")
    return events

def main():
    if len(sys.argv) < 2:
        print("Usage: python yolo_analyzer.py <video.mp4>")
        sys.exit(1)
    analyze_gameplay(sys.argv[1])


if __name__ == "__main__":
    main()
