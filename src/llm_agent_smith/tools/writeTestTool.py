import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

from llm_agent_smith.states.TDDState import TDDState
from llm_agent_smith.models.geminiModel import GeminiModel

llm = GeminiModel.llm_model()


def extract_code(text: str) -> str:
    """Extrai código de blocos markdown"""
    if "```python" in text:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


def write_test(state: TDDState) -> TDDState:
    """Escreve um teste falhando para a feature atual"""
    prompt = ChatPromptTemplate.from_template(
        "Escreva um teste Pytest para a feature:\n{feature}\n\n"
        "Contexto:\nCódigo atual:\n{code}\n\n"
        "Diretrizes:\n- Teste apenas o essencial\n- Espere falhar inicialmente\n"
        "Código do teste:"
    )

    chain = prompt | llm
    response = chain.invoke(
        {"feature": state["current_feature"], "code": state["production_code"]}
    )

    content = getattr(response, "content", str(response))

    new_test = extract_code(content)
    updated_test_code = state["test_code"] + "\n\n" + new_test

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "Escrever teste (Fase RED)",
        "details": new_test[:500] + "..." if len(new_test) > 500 else new_test,
    }

    print(
        f"📝 Teste escrito (Fase RED):\n{new_test[:200]}{'...' if len(new_test) > 200 else ''}"
    )

    return {
        **state,
        "test_code": updated_test_code,
        "history": state["history"] + [history_entry],
        "iteration_count": state["iteration_count"] + 1,
    }
