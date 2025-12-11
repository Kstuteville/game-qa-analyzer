"""
llm_agent.py

Generates a UX / QA analysis report using the OpenAI API.
"""

import json
import os
from openai import OpenAI

# Read API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_ux_report(merged_log_path: str, game_context: str) -> str:
    """
    Generate a UX research report based on merged gameplay + emotion logs.
    """

    # Load merged timeline
    with open(merged_log_path, "r") as f:
        merged_data = json.load(f)

    # Limit preview for token safety
    preview_data = merged_data[:50]

    prompt = f"""
You are a senior Game UX Researcher and QA Analyst.

You are given:
1. GAME CONTEXT:
{game_context}

2. MERGED GAMEPLAY + EMOTION LOG (first 50 entries):
{json.dumps(preview_data, indent=2)}

Write a structured UX/QA analysis identifying:
- Moments of frustration
- Player confusion or hesitation
- High cognitive load moments
- Repeated failure loops
- Poorly communicated mechanics or unclear feedback
- Correlations between gameplay events and emotional states
- Actionable recommendations for UI, level design, onboarding, and feedback clarity

Write clearly in sections, as if delivering a real studio report.
Do NOT mention AI.
    """

    # Call OpenAI Responses API
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    # Return report text
    return response.output_text


# Optional debug run
if __name__ == "__main__":
    print("This file is not meant to be run directly.")
