"""
llm_agent.py

Generates a UX / QA analysis report using the OpenAI API.
Works on both Mac and Dell environments.
"""

import json
import os
from openai import OpenAI

# Read API key from environment variable
# Make sure you run: export OPENAI_API_KEY="your_key_here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ux_report(merged_log_path: str, game_context: str) -> str:
    """
    Generate an AI-powered UX report based on merged gameplay+emotion logs.

    Args:
        merged_log_path: Path to merged_log.json
        game_context: Text description of the gameplay scenario

    Returns:
        str - A UX analysis report
    """

    # Load merged timeline
    with open(merged_log_path, "r") as f:
        merged_data = json.load(f)

    # Only send first 50 entries (safe for tokens)
    preview_data = merged_data[:50]

    prompt = f"""
You are a senior Game UX Researcher and QA Analyst.

You are given:
1. GAME CONTEXT:
{game_context}

2. MERGED GAMEPLAY + EMOTION LOG (first 50 entries):
{json.dumps(preview_data, indent=2)}

Your goal:
Write a structured UX/QA analysis identifying:
- Moments of frustration
- Player confusion or hesitation
- High cognitive load moments
- Repeated failure loops
- Poorly communicated mechanics or unclear feedback
- Where emotion state and gameplay events correlate
- Actionable design recommendations for UI, level design, and tutorial flow

Write clearly, in sections, as if delivering a real studio report.
Do NOT apologize or mention AI.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    # Extract output text
    return response.output_text
