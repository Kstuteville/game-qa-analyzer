"""
yolo_analyzer.py — Stable Mac + Dell Version
Safe for CPU-only and GPU environments.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def analyze_gameplay(video_path: str, output_json_path: str = "gameplay_log.json") -> list:
    events = []

    # Load YOLOv8 model
    print("Loading YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")  # auto-downloads if missing

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Total Frames: {total_frames}")

    FRAME_SAMPLE_RATE = 3
    DARKNESS_THRESHOLD = 30
    STAGNATION_THRESHOLD = 0.02
    STAGNATION_DURATION = 2.0

    ENEMY_CLASSES = {'person', 'dog', 'cat', 'horse', 'bear', 'zebra', 'giraffe', 'elephant', 'bird'}
    PROJECTILE_CLASSES = {'sports ball', 'frisbee', 'kite', 'baseball bat', 'baseball glove'}
    WEAPON_CLASSES = {'knife', 'baseball bat', 'tennis racket', 'sports ball'}

    frame_idx = 0
    prev_frame_gray = None
    stagnation_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        if frame_idx % FRAME_SAMPLE_RATE == 0:

            # grayscale frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(frame_gray))

            # Run YOLO detection
            try:
                results = model.predict(frame, verbose=False)
            except Exception as e:
                print(f"YOLO error on frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            detected_objects = []
            enemy_count = 0
            projectile_count = 0
            weapon_count = 0

            # Parse YOLO results
            if len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())  # safer item()
                        class_name = model.names.get(cls_id, "unknown")
                        confidence = float(box.conf.item())

                        if confidence > 0.5:
                            detected_objects.append(class_name)

                            if class_name in ENEMY_CLASSES:
                                enemy_count += 1
                            if class_name in PROJECTILE_CLASSES:
                                projectile_count += 1
                            if class_name in WEAPON_CLASSES:
                                weapon_count += 1

            # Log simple events
            if enemy_count > 0:
                events.append({
                    "time": round(timestamp, 2),
                    "event": "enemy",
                    "details": {
                        "count": enemy_count,
                        "brightness": round(mean_brightness, 2)
                    }
                })

            if projectile_count > 0:
                events.append({
                    "time": round(timestamp, 2),
                    "event": "projectile",
                    "details": {"count": projectile_count}
                })

            if weapon_count > 0:
                events.append({
                    "time": round(timestamp, 2),
                    "event": "weapon_like",
                    "details": {"count": weapon_count}
                })

            if enemy_count >= 3:
                events.append({
                    "time": round(timestamp, 2),
                    "event": "combat_spike",
                    "details": {"count": enemy_count}
                })

            # Death/loading detection
            if len(detected_objects) == 0 and mean_brightness < DARKNESS_THRESHOLD:
                events.append({
                    "time": round(timestamp, 2),
                    "event": "death_or_loading_screen",
                    "details": {"brightness": round(mean_brightness, 2)}
                })

            # Stagnation detection
            if prev_frame_gray is not None:
                diff = cv2.absdiff(frame_gray, prev_frame_gray)
                diff_norm = float(np.mean(diff)) / 255.0

                if diff_norm < STAGNATION_THRESHOLD:
                    if stagnation_start_time is None:
                        stagnation_start_time = timestamp
                    elif (timestamp - stagnation_start_time) >= STAGNATION_DURATION:
                        events.append({
                            "time": round(timestamp, 2),
                            "event": "stagnation_or_pause",
                            "details": {"duration": round(timestamp - stagnation_start_time, 2)}
                        })
                else:
                    stagnation_start_time = None

            prev_frame_gray = frame_gray.copy()

        frame_idx += 1

        # Optional lightweight progress update
        if frame_idx % 300 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    # Save log
    with open(output_json_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nSaved gameplay analysis to {output_json_path}")
    return events


def main():
    if len(sys.argv) < 2:
        print("Usage: python yolo_analyzer.py <video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    analyze_gameplay(video_path)


if __name__ == "__main__":
    main()
