from langchain_google_genai import ChatGoogleGenerativeAI
from config.main import AppConfig


class GeminiModel:
    @staticmethod
    def llm_model():
        return ChatGoogleGenerativeAI(model=AppConfig.DEFAULT_LLM_MODEL)
