# 🧠 LLM Agent Smith

Agente LLM autoestruturante com arquitetura hierárquica delegatória, usando LangGraph e Poetry.

## 🎯 Objetivo

Criar um sistema de agentes que:
- Decodifica comandos textuais
- Constrói um plano hierárquico da solução
- Divide responsabilidades e delega a outros agentes especialistas
- Utiliza ciclos de feedback com validação TDD até a entrega da solução final

## 📦 Tecnologias

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Poetry](https://python-poetry.org/)
- [Python 3.11+](https://www.python.org/)
- [Vector DB opcional](https://github.com/weaviate/weaviate) (e.g. FAISS, ChromaDB)

🚀 Preparando o projeto
```bash
  git clone https://github.com/natanrelva/llm_agent_smith
  cd llm-agent-smith
  poetry install
  poetry env use python3.11 
  source $(poetry env info --path)/bin/activate
```
🧪 Testes
```bash
poetry run pytest
```

## 🏗️ Arquitetura

```mermaid
graph TD
  A[Usuário envia entrada] --> B[Agente Decodificador]
  B --> C[Cria Plano Hierárquico]
  C --> D[Armazena em Memória Vetorial]
  C --> E[Agente Delegatório]
  E --> F[Criação de subagentes com responsabilidade única]
  F --> G[Testes automatizados e ciclos de feedback]
  G --> H[Agente Integrador]
  H --> I[Resposta final entregue]
