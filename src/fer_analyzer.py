"""
fer_analyzer.py
Robust emotion detection using FER (Facial Emotion Recognition).
Works reliably on real facecam video for demos.
"""


import json
import numpy as np
from pathlib import Path
import cv2
from fer import FER

def analyze_face(video_path: str, output_json_path="emotion_log.json"):
   video_path = Path(video_path)
   if not video_path.exists():
       raise ValueError(f"Face video not found: {video_path}")
   print("Running FER emotion analysis…")
   cap = cv2.VideoCapture(str(video_path))
   fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
   frame_num = 0
   detector = FER(mtcnn=False)  # use CV-based detection for reliability
   entries = []
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       time = round(frame_num / fps, 2)
       frame_num += 1
       # FER detection
       try:
           detections = detector.detect_emotions(frame)
       except Exception as e:
           print(f"FER error at time {time}: {e}")
           continue
       if len(detections) == 0:
           # No face detected → fallback unknown entry
           entries.append({
               "time": time,
               "raw_emotion": "unknown",
               "emotion_state": "unknown",
               "confidence": 0.0
           })
           continue
       # FER may return multiple faces; take the first
       face_data = detections[0]["emotions"]
       # Select emotion with highest probability
       emotion = max(face_data, key=face_data.get)
       conf = float(face_data[emotion])
       entries.append({
           "time": time,
           "raw_emotion": emotion,
           "emotion_state": emotion,
           "confidence": round(conf, 3)
       })
   cap.release()
   # Fallback if no valid entries at all
   if len(entries) == 0:
       print("⚠️ No emotions detected — writing fallback neutral entry.")
       entries = [{
           "time": 0.0,
           "raw_emotion": "unknown",
           "emotion_state": "unknown",
           "confidence": 0.0
       }]
   # Save the log
   with open(output_json_path, "w") as f:
       json.dump(entries, f, indent=2)
   print(f"Emotion log saved → {output_json_path}")
   print(f"Detected {len(entries)} frames.")
   return entries
# Debug entry point
if __name__ == "__main__":
   import sys
   analyze_face(sys.argv[1])
