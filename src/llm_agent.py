"""
llm_agent.py

Generates a UX / QA analysis report using the OpenAI API.
Uses full-session coverage via sampling + "top moments" selection.
"""

import json
import os
from collections import Counter
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Sampling helpers
# -----------------------------
def sample_evenly(data: list, k: int) -> list:
    """Pick k items spread evenly across the whole list."""
    if not data:
        return []
    if len(data) <= k:
        return data
    idxs = [int(i * (len(data) - 1) / (k - 1)) for i in range(k)]
    return [data[i] for i in idxs]


def pick_interesting_moments(data: list, k: int = 40) -> list:
    """
    Pick up to k entries that are likely meaningful:
    - troop_spike, building_spike, spell_cast
    - emotion changes
    - stagnation_or_pause
    """
    if not data:
        return []

    interesting = []
    prev_emo = None

    for e in data:
        ev = e.get("game_event", "")
        emo = e.get("emotion_state", "")

        # high-signal events
        if ev in {"troop_spike", "building_spike", "spell_cast", "stagnation_or_pause"}:
            interesting.append(e)
            prev_emo = emo
            continue

        # emotion change moments
        if prev_emo is None:
            prev_emo = emo
        elif emo != prev_emo:
            interesting.append(e)
            prev_emo = emo

    # if we collected too many, downsample evenly
    return sample_evenly(interesting, min(k, len(interesting)))

def unique_detected_cards(merged_data: list) -> dict:
    troops=set(); spells=set(); buildings=set()
    for e in merged_data:
        d = e.get("game_details", {}) or {}
        for t in d.get("troops", []) or []: troops.add(t)
        for s in d.get("spells", []) or []: spells.add(s)
        for b in d.get("buildings", []) or []: buildings.add(b)
    return {
        "troops": sorted(troops),
        "spells": sorted(spells),
        "buildings": sorted(buildings)
    }


def build_session_summary(data: list) -> dict:
    """Lightweight stats to give the LLM context without raw frames."""
    if not data:
        return {"note": "No merged data found."}

    event_counts = Counter()
    emotion_counts = Counter()
    top_troops = Counter()
    top_spells = Counter()

    max_time = 0.0
    for e in data:
        max_time = max(max_time, float(e.get("time", 0)))
        event_counts[e.get("game_event", "unknown")] += 1
        emotion_counts[e.get("emotion_state", "unknown")] += 1

        details = e.get("game_details", {}) or {}
        # your yolo_analyzer puts troops/spells as lists inside game_details
        for t in details.get("troops", []) or []:
            top_troops[t] += 1
        for s in details.get("spells", []) or []:
            top_spells[s] += 1

    return {
        "duration_sec_est": round(max_time, 2),
        "event_counts_top": dict(event_counts.most_common(15)),
        "emotion_counts": dict(emotion_counts.most_common(15)),
        "top_troops": dict(top_troops.most_common(20)),
        "top_spells": dict(top_spells.most_common(20)),
        "note": "Counts are based on sampled/logged events, not ground-truth game telemetry."
    }


# -----------------------------
# Main report generator
# -----------------------------
def generate_ux_report(merged_log_path: str, game_context: str) -> str:
    with open(merged_log_path, "r") as f:
        merged_data = json.load(f)

    # ✅ cover full session without huge tokens
    overview = build_session_summary(merged_data)
    evenly_sampled = sample_evenly(merged_data, k=140)
    interesting = pick_interesting_moments(merged_data, k=60)
    coverage = unique_detected_cards(merged_data)


    payload = {
        "overview": overview,
        "evenly_sampled_entries": evenly_sampled,
        "interesting_moments": interesting,
    }

    prompt = f"""
You are a senior Game UX Researcher and QA Analyst.

GAME CONTEXT:
{game_context}

You are given a merged gameplay+emotion session.
- `game_event` describes the event type.
- `game_details` contains specific detected troop/building/spell names (use them explicitly).

REQUIRED COVERAGE LIST:
You MUST mention every item in these lists at least once in the report (even briefly),
or explicitly say "not enough evidence to analyze" for that item.

SESSION DATA (summary + sampled timeline):
{json.dumps(payload, indent=2)}

Write a structured UX/QA analysis:
- Where the player seems engaged vs disengaged
- Troop/spell moments correlated with emotion shifts
- Moments of hesitation (stagnation/pause)
- Actionable recommendations (UI feedback, pacing, onboarding, clarity)

Be specific: name troops/spells when present in `game_details`.
Do NOT mention AI.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":
    print("This file is not meant to be run directly.")
