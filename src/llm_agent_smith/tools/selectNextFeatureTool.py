from datetime import datetime

from llm_agent_smith.states.TDDState import TDDState


def select_next_feature(state: TDDState) -> TDDState:
    """Seleciona a próxima feature a ser implementada"""
    if not state["features"]:
        return {**state, "current_feature": None}

    next_feature = state["features"].pop(0)

    history_entry = {
        "timestamp": datetime.now(),
        "action": "Selecionar feature",
        "details": next_feature,
    }

    return {
        **state,
        "current_feature": next_feature,
        "iteration_count": 0,
        "history": state["history"] + [history_entry],
    }
