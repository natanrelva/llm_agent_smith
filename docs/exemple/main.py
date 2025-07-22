import asyncio
import platform
import shutil
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator
import re
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile
from llm_agent_smith.models.geminiModel import GeminiModel

llm_model = GeminiModel.llm_model()
BASE_DIR = Path(__file__).parent / "tdd_projects"
BASE_DIR.mkdir(exist_ok=True)


class TDDState(TypedDict):
    task: str
    code: str
    tests: str
    test_results: str
    feedback: List[str]
    iteration: Annotated[int, operator.add]
    project_dir: Path


test_prompt = ChatPromptTemplate.from_template(
    "Você é um engenheiro de software especializado em TDD. Para a tarefa: '{task}'\n"
    "Gere um conjunto de testes unitários em Python usando unittest que cubram:\n"
    "1. Casos de uso principais (ex.: strings comuns, números inteiros)\n"
    "2. Casos extremos (ex.: strings vazias, nulas, com acentos, caracteres especiais)\n"
    "3. Tratamento de erros (ex.: entradas inválidas como None ou tipos incorretos)\n\n"
    "Formato de resposta:\n"
    "```python\n"
    "import unittest\n"
    "from solution import *\n\n"
    "class TestSolution(unittest.TestCase):\n"
    "    # Testes devem ser claros, concisos e cobrir todos os casos mencionados\n"
    "```"
)

code_prompt = ChatPromptTemplate.from_template(
    "Com base nos testes abaixo, implemente a solução em Python para a tarefa: {task}\n\n"
    "Testes:\n{tests}\n\n"
    "Feedback do último ciclo: {feedback}\n\n"
    "Instruções:\n"
    "- Implemente APENAS o código necessário para passar nos testes\n"
    "- Siga as melhores práticas de clean code (nomes claros, código simples)\n"
    "- Considere casos extremos (strings vazias, acentos, caracteres especiais)\n"
    "- Valide entradas para evitar erros (ex.: None, tipos incorretos)\n"
    "- Inclua comentários apenas se necessário para clareza\n\n"
    "Código:"
)

refactor_prompt = ChatPromptTemplate.from_template(
    "Analise o código e os resultados dos testes para a tarefa: {task}\n\n"
    "Código:\n{code}\n\n"
    "Testes:\n{tests}\n\n"
    "Resultados dos Testes:\n{test_results}\n\n"
    "Forneça feedback detalhado para:\n"
    "1. Correção de erros (identifique falhas específicas e sugira correções)\n"
    "2. Sugestões de refatoração (melhorar legibilidade, estrutura, modularidade)\n"
    "3. Melhorias de performance (otimizações, se aplicável)\n"
    "4. Cobertura adicional de testes (casos faltantes, como acentos ou erros)\n\n"
    "Feedback:"
)


def create_project_dir(task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_slug = re.sub(r"\W+", "_", task)[:50].strip("_")
    project_name = f"{timestamp}_{task_slug}"
    project_dir = BASE_DIR / project_name
    project_dir.mkdir(exist_ok=True)
    (project_dir / "versions").mkdir(exist_ok=True)
    return project_dir


def save_code_version(state: TDDState, iteration: int):
    project_dir = state["project_dir"]
    version_dir = project_dir / "versions" / f"iter_{iteration}"
    version_dir.mkdir(exist_ok=True)
    with open(version_dir / "solution.py", "w") as f:
        f.write(state["code"])
    with open(version_dir / "test_solution.py", "w") as f:
        f.write(state["tests"])
    with open(version_dir / "test_results.txt", "w") as f:
        f.write(state["test_results"])
    with open(version_dir / "feedback.txt", "w") as f:
        f.write("\n\n".join(state["feedback"]))


def save_final_version(state: TDDState):
    project_dir = state["project_dir"]
    with open(project_dir / "solution.py", "w") as f:
        f.write(state["code"])
    with open(project_dir / "test_solution.py", "w") as f:
        f.write(state["tests"])
    readme_content = f"""# TDD Agent Project\n\n## Task: {state["task"]}\n\n### Summary\n- Iterations: {state["iteration"]}\n- Final Test Status: {"PASS" if "OK" in state["test_results"] else "FAIL"}\n\n### Final Feedback:\n{state["feedback"][-1] if state["feedback"] else "No feedback"}\n\n## How to Run\n```bash\npython test_solution.py\n```"""
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_content)


def generate_tests(state: TDDState):
    task = state["task"]
    messages = test_prompt.format_messages(task=task)
    response = llm_model.invoke(messages)
    tests = response.content.strip()
    if "```python" in tests:
        tests = re.search(r"```python(.*?)```", tests, re.DOTALL).group(1).strip()
    return {"tests": tests}


def generate_code(state: TDDState):
    task = state["task"]
    tests = state["tests"]
    feedback = (
        "\n".join(state["feedback"][-1:])
        if state["feedback"]
        else "Nenhum feedback ainda"
    )
    messages = code_prompt.format_messages(task=task, tests=tests, feedback=feedback)
    response = llm_model.invoke(messages)
    code = response.content.strip()
    if "```python" in code:
        code = re.search(r"```python(.*?)```", code, re.DOTALL).group(1).strip()
    return {"code": code}


def analyze_and_refactor(state: TDDState):
    task = state["task"]
    code = state["code"]
    tests = state["tests"]
    test_results = state["test_results"]
    messages = refactor_prompt.format_messages(
        task=task, code=code, tests=tests, test_results=test_results
    )
    response = llm_model.invoke(messages)
    feedback = response.content.strip()
    return {"feedback": state["feedback"] + [feedback], "iteration": 1}


def should_continue(state: TDDState):
    if state["iteration"] >= 5:
        return "end"
    test_output = state["test_results"]
    if "OK" in test_output and (
        "FAILED" not in test_output and "ERROR" not in test_output
    ):
        return "end"
    last_feedback = state["feedback"][-1].lower() if state["feedback"] else ""
    if (
        "todos os testes passaram" in last_feedback
        or "não há mais melhorias" in last_feedback
    ):
        return "end"
    return "continue"


def open_project_dir(project_dir: Path, platform_system: str = platform.system()):
    try:
        if platform_system == "win32":
            os.startfile(project_dir)
        elif platform_system == "darwin":
            subprocess.run(["open", str(project_dir)], check=False)
        else:
            if shutil.which("xdg-open"):
                subprocess.run(["xdg-open", str(project_dir)], check=False)
            else:
                print(
                    f"⚠️ 'xdg-open' não encontrado. Acesse o diretório manualmente em: {project_dir}"
                )
    except Exception as e:
        print(f"⚠️ Erro ao abrir o diretório: {e}. Acesse manualmente em: {project_dir}")


def run_unit_tests(code: str, tests: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        solution_path.write_text(code)
        test_path = tmp_path / "test_solution.py"
        test_path.write_text(tests)
        (tmp_path / "__init__.py").touch()
        print(f"\n📜 Código a ser testado:\n{code}\n")
        print(f"📋 Testes a serem executados:\n{tests}\n")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "test_solution", "-v"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return "ERRO: Timeout ao executar testes"
        except Exception as e:
            return f"ERRO: {str(e)}"


def execute_tests(state: TDDState):
    test_results = run_unit_tests(state["code"], state["tests"])
    save_code_version(state, state["iteration"])
    return {"test_results": test_results}


workflow = StateGraph(TDDState)
workflow.add_node("gerar_testes", generate_tests)
workflow.add_node("gerar_codigo", generate_code)
workflow.add_node("executar_testes", execute_tests)
workflow.add_node("analisar_refatorar", analyze_and_refactor)
workflow.set_entry_point("gerar_testes")
workflow.add_edge("gerar_testes", "gerar_codigo")
workflow.add_edge("gerar_codigo", "executar_testes")
workflow.add_edge("executar_testes", "analisar_refatorar")
workflow.add_conditional_edges(
    "analisar_refatorar", should_continue, {"continue": "gerar_codigo", "end": END}
)
tdd_agent = workflow.compile()


def run_tdd_agent(task: str, max_iterations: int = 5, open_dir: bool = True):
    project_dir = create_project_dir(task)
    print(f"📂 Projeto salvo em: {project_dir}")
    inputs = {
        "task": task,
        "code": "",
        "tests": "",
        "test_results": "",
        "feedback": [],
        "iteration": 0,
        "project_dir": project_dir,
    }
    print(f"🧪 Iniciando ciclo TDD para: {task}")
    print("=" * 60)

    current_state = inputs
    for output in tdd_agent.stream(inputs):
        for node, partial_state in output.items():
            if node == "__end__":
                final_state = current_state
                continue
            current_state = {**current_state, **partial_state}
            iteration = current_state["iteration"]
            print(f"\n🔁 ITERAÇÃO {iteration} - {node.upper()}")
            print("-" * 50)
            if node == "gerar_testes":
                print("📋 TESTES GERADOS:")
                print(current_state["tests"])
            elif node == "gerar_codigo":
                print("💻 CÓDIGO IMPLEMENTADO:")
                print(current_state["code"])
            elif node == "executar_testes":
                print("✅ RESULTADOS DOS TESTES:")
                print(current_state["test_results"])
            elif node == "analisar_refatorar":
                print("🔍 FEEDBACK DE REFATORAÇÃO:")
                print(current_state["feedback"][-1])

    final_state = current_state
    save_final_version(final_state)

    print("\n" + "=" * 60)
    print("🏁 CICLO TDD CONCLUÍDO!")
    print(f"📂 Projeto final salvo em: {project_dir}")
    print(f"⚙️ Total de iterações: {final_state['iteration']}")
    test_output = final_state["test_results"]
    if "OK" in test_output and "FAILED" not in test_output:
        print("✅ TODOS OS TESTES PASSARAM!")
    else:
        print(f"⚠️ PROBLEMAS NOS TESTES: {test_output.splitlines()[-1]}")
    if open_dir:
        open_project_dir(project_dir)
    return final_state


async def main():
    task = (
        "Implemente uma classe StringAnalyzer com métodos para:\n"
        "1. Contar vogais (incluindo acentos, ex.: á, é)\n"
        "2. Contar consoantes (excluindo caracteres especiais)\n"
        "3. Inverter string (preservando a codificação)\n"
        "4. Verificar palíndromo (ignorando espaços, pontuação e maiúsculas/minúsculas)"
    )
    result = run_tdd_agent(task, open_dir=True)
    return result


if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())
