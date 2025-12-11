
 
import sys
import json
from pathlib import Path
from typing import List, Dict, Union


def load_json(path: Union[str, Path]) -> list:
    """Load a JSON list safely."""
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
) -> List[Dict]:

    gameplay = load_json(gameplay_log_path)
    emotion = load_json(emotion_log_path)

    print(f"Loaded {len(gameplay)} gameplay events")
    print(f"Loaded {len(emotion)} emotion entries")

    # Sort by time
    gameplay.sort(key=lambda x: x.get("time", 0))
    emotion.sort(key=lambda x: x.get("time", 0))

    # If no emotion data, fallback gracefully
    if len(emotion) == 0:
        print("⚠️ Warning: Emotion log is empty. All merged records will use emotion='unknown'")
        merged = [
            {
                "time": ev.get("time", 0),
                "game_event": ev.get("event", "unknown"),
                "emotion_state": "unknown",
                "game_details": ev.get("details", {})
            }
            for ev in gameplay
        ]
        with open(output_json_path, "w") as f:
            json.dump(merged, f, indent=2)
        return merged

    merged = []
    emo_idx = 0
    emo_len = len(emotion)

    # Linear-time merge O(N)
    for event in gameplay:
        t = event.get("time", 0)

        # Advance emotion pointer while next emotion timestamp is closer
        while (
            emo_idx < emo_len - 1 and
            abs(emotion[emo_idx + 1].get("time", 0) - t)
            < abs(emotion[emo_idx].get("time", 0) - t)
        ):
            emo_idx += 1

        # Now emotion[emo_idx] is the closest timestamp
        closest = emotion[emo_idx]
        diff = abs(closest.get("time", 0) - t)

        if diff <= max_time_diff:
            # Updated FER schema: "emotion_state"
            emotion_state = closest.get("emotion_state", "unknown")
        else:
            emotion_state = "unknown"

        merged.append({
            "time": t,
            "game_event": event.get("event", "unknown"),
            "emotion_state": emotion_state,
            "confidence": closest.get("confidence", None)
                if diff <= max_time_diff else None,
            "game_details": event.get("details", {})
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
