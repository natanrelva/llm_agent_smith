import re
from langgraph.graph import StateGraph
from llm_agent_smith.models.geminiModel import GeminiModel
from llm_agent_smith.states.TDDState import TDDState
from llm_agent_smith.tools.decomposeFeaturesTool import decompose_features
from llm_agent_smith.tools.selectNextFeatureTool import select_next_feature
from datetime import datetime

from llm_agent_smith.tools.writeTestTool import write_test


def run_tests(production_code: str, test_code: str) -> str:
    """Executa os testes e retorna os resultados"""
    if not test_code.strip():
        return "Nenhum teste definido"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Salvar código de produção
        prod_path = Path(tmpdir) / "production.py"
        prod_path.write_text(production_code)

        # Adicionar imports necessários
        test_content = (
            "import re\nimport sys\nsys.path.insert(0, '.')\nfrom production import *\n\n"
            + test_code
        )

        # Salvar testes
        test_path = Path(tmpdir) / "test_production.py"
        test_path.write_text(test_content)

        try:
            result = subprocess.run(
                ["pytest", "-v", str(test_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = "ERRO: Timeout ao executar testes"
        except Exception as e:
            output = f"ERRO: {str(e)}"

        return output


def execute_tests(state: TDDState) -> TDDState:
    """Executa os testes e armazena os resultados"""
    test_results = run_tests(state["production_code"], state["test_code"])

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "Executar testes",
        "details": (
            test_results[:500] + "..." if len(test_results) > 500 else test_results
        ),
    }

    print(
        f"🧪 Resultado dos testes:\n{test_results[:300]}{'...' if len(test_results) > 300 else ''}"
    )

    return {
        **state,
        "test_results": test_results,
        "history": state["history"] + [history_entry],
    }


graph = StateGraph(TDDState)
graph.add_node("decompose_features", decompose_features)
graph.add_node("select_next_feature", select_next_feature)
graph.add_node("write_test", write_test)

graph.set_entry_point("decompose_features")
graph.add_edge("decompose_features", "select_next_feature")
graph.add_edge("select_next_feature", "write_test")

graph.set_finish_point("write_test")

tdd_app = graph.compile()

if __name__ == "__main__":
    initial_state: TDDState = {
        "user_request": "Implemente um validador de CPF que verifique formato e dígitos",
        "features": [],
        "current_feature": None,
        "production_code": "",
        "test_code": "",
        "test_results": None,
        "history": [],
        "iteration_count": 0,
    }
    final_state = tdd_app.invoke(initial_state)
    for h in final_state["history"]:
        print(f"- [{h['timestamp']}] {h['action']}: {h['details']}")
