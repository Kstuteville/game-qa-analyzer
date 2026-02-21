import sys
import json
from pathlib import Path
from typing import List, Dict, Union, Any


def load_json(path: Union[str, Path]) -> list:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain an array, found {type(data).__name__}")
    return data

def merge_logs(
    gameplay_log_path: str = "gameplay_log.json",
    emotion_log_path: str = "emotion_log.json",
    output_json_path: str = "merged_log.json",
    max_time_diff: float = 2.0
) -> List[Dict[str, Any]]:
    gameplay = load_json(gameplay_log_path)
    emotion = load_json(emotion_log_path)
    print(f"Loaded {len(gameplay)} gameplay events")
    print(f"Loaded {len(emotion)} emotion entries")
    gameplay.sort(key=lambda x: x.get("time", 0))
    emotion.sort(key=lambda x: x.get("time", 0))
    merged: List[Dict[str, Any]] = []
    emo_idx = 0
    emo_len = len(emotion)
    for event in gameplay:
        t = float(event.get("time", 0))
        while (
            emo_idx < emo_len - 1 and
            abs(float(emotion[emo_idx + 1].get("time", 0)) - t)
            < abs(float(emotion[emo_idx].get("time", 0)) - t)
        ):
            emo_idx += 1
        closest = emotion[emo_idx]
        diff = abs(float(closest.get("time", 0)) - t)
        if diff <= max_time_diff:
            emotion_state = (
                closest.get("emotion_state") or
                closest.get("raw_emotion") or
                closest.get("emotion") or
                "unknown"
            )
            confidence = closest.get("confidence", None)
        else:
            emotion_state = "unknown"
            confidence = None
        game_event = event.get("game_event") or event.get("event", "unknown")
        game_details = event.get("details", {}) if isinstance(event.get("details", {}), dict) else {}
        merged.append({
            "time": t,
            "game_event": game_event,
            "game_details": game_details,   
            "emotion_state": emotion_state,
            "confidence": confidence,
        })

    with open(output_json_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged log saved → {output_json_path}")
    return merged

def main():
    gameplay_path = sys.argv[1] if len(sys.argv) > 1 else "gameplay_log.json"
    emotion_path = sys.argv[2] if len(sys.argv) > 2 else "emotion_log.json"
    output = sys.argv[3] if len(sys.argv) > 3 else "merged_log.json"
    max_diff = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    merge_logs(gameplay_path, emotion_path, output, max_diff)


if __name__ == "__main__":
    main()
