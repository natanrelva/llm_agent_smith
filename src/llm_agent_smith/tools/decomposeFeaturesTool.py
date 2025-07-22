import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from llm_agent_smith.states.TDDState import TDDState
from llm_agent_smith.models.geminiModel import GeminiModel

llm = GeminiModel.llm_model()


def decompose_features(state: TDDState) -> TDDState:
    prompt = ChatPromptTemplate.from_template(
        "Solicitação do usuário: {request}\n\n"
        "Decomponha em features mínimas testáveis (MFVs):\n"
        "- Uma por linha\n"
        "- Ordenadas por dependência\n"
        "- Formato JSON array\n\n"
        "Apenas a lista em formato JSON:"
    )
    chain = prompt | llm
    response = chain.invoke({"request": state["user_request"]})
    content = getattr(response, "content", str(response))
    try:
        features = json.loads(str(content))
    except Exception:
        features = [state["user_request"]]
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "Decomposição de features",
        "details": str(content)[:500],
    }
    return {
        **state,
        "features": features,
        "history": state["history"] + [history_entry],
    }
