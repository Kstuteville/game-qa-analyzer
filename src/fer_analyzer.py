"""
fer_analyzer.py
Uses the FER library (v22.5.0) for facial emotion detection on video.
DEPENDS ONLY ON: fer, cv2, json, numpy
"""

import json
from pathlib import Path
from fer import FER, Video

def analyze_face(video_path: str, output_json_path: str = "emotion_log.json") -> list:
    """
    Analyze a face video and extract emotion data using FER (legacy API).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError(f"Face video not found: {video_path}")
    detector = FER(mtcnn=True)
    video = Video(str(video_path))
    print("Running FER emotion analysis...")
    raw_data = video.analyze(detector, display=False)
    entries = []
    for frame in raw_data:
        if "emotions" not in frame:
            continue
        time = round(frame["time"], 2)
        emotions = frame["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = float(emotions[dominant])
        if dominant in ["angry", "sad"]:
            state = "frustrated"
        elif dominant in ["fear", "surprise"]:
            state = "confused"
        elif dominant == "happy":
            state = "positive"
        else:
            state = "focused"

        entries.append({
            "time": time,
            "raw_emotion": dominant,
            "state": state,
            "confidence": round(confidence, 3),
        })
    with open(output_json_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Emotion log saved to {output_json_path}")
    return entries
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fer_analyzer.py <face_video>")
        exit()

    analyze_face(sys.argv[1])
