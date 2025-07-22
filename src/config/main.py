import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    DEFAULT_LLM_MODEL = os.getenv(
        "DEFAULT_LLM_MODEL", "gemini-2.5-flash"
    )  # Permite override via .env

    # Você pode adicionar métodos para validação ou outras configurações complexas
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY não encontrada no arquivo .env ou variáveis de ambiente."
            )
        print("Configurações do ambiente carregadas e validadas com sucesso.")


# Chama a validação assim que o módulo é carregado
# Isso garante que as variáveis críticas existam ANTES de qualquer outro código usar AppConfig
AppConfig.validate()
