import subprocess
import json

def explain(decision):

    prompt = f"""
    You are a financial analyst.
    Explain this market outlook in simple language:

    {json.dumps(decision, indent=2)}
    """
    result = subprocess.run(
        ["ollama", "run", "mistral", prompt],
        capture_output=True,
        text=True
    )

    return result.stdout
