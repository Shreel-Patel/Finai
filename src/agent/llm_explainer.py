import subprocess
import json

def generate_explanation(agent_output, market_snapshot):
    prompt = f"""
You are a professional financial market analyst.

Return ONLY plain text using EXACTLY this structure.
Do not use markdown, emojis, or extra commentary.

FORMAT:

MARKET_SUMMARY:
- sentence
- sentence

AI_REASONING:
- sentence
- sentence

RISKS:
- sentence
- sentence

Market snapshot:
{json.dumps(market_snapshot, indent=2)}

Agent decision:
{json.dumps(agent_output, indent=2)}
"""

    result = subprocess.run(
        ["ollama", "run", "gemma:2b", prompt],
        capture_output=True,
        text=True
    )

    return parse_llm_output(result.stdout.strip())


def parse_llm_output(text: str):
    sections = {
        "market_summary": [],
        "ai_reasoning": [],
        "risks": []
    }

    current = None
    for line in text.splitlines():
        line = line.strip()

        # Handle both formats: with and without ##
        if "MARKET_SUMMARY" in line:
            current = "market_summary"
        elif "AI_REASONING" in line:
            current = "ai_reasoning"
        elif "RISKS" in line:
            current = "risks"
        elif line.startswith("-") and current:
            # Remove the dash and any extra whitespace
            content = line[1:].strip()
            if content:  # Only add if not empty
                sections[current].append(content)

    return sections