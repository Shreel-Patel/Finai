import pandas as pd
from pathlib import Path
from src.agent.agent import run_agent

_root = Path(__file__).resolve().parent.parent.parent
# Load final dataset
df = pd.read_csv(_root / "data" / "final" / "AAPL.csv")

# Pick the latest valid row
row = df.dropna().iloc[-1]

# Fake model output (for now)
row = row.copy()
row["prob_up"] = 0.48   # simulate model confidence

# Run agent
decision = run_agent(row)

print("\n=== AGENT OUTPUT ===")
for k, v in decision.items():
    print(f"{k}: {v}")
