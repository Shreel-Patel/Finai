import pandas as pd
from src.agent.agent import run_agent

# Load final dataset
df = pd.read_csv("C:\\Users\\SHREEL\\PycharmProjects\\FINAI\\data\\final\\AAPL.csv")

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
