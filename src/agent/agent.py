from src.agent.state_builder import build_state
from src.agent.decision_rules import decide
def run_agent(row):
    state = build_state(row)
    decision = decide(state)

    decision["prob_up"] = round(state["prob_up"], 3)
    return decision
