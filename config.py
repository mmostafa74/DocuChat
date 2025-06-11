import os
import toml
from dotenv import load_dotenv
import warnings

load_dotenv()


class Config:
    def __init__(self, config_file: str = "config.toml"):
        with open(config_file, "r") as f:
            self._config = toml.load(f)
        self._override_with_env()
        self._validate()

    def _override_with_env(self):
        self._config.setdefault("app", {})
        self._config.setdefault("ui", {})
        self._config.setdefault("sidebar", {})
        self._config.setdefault("styles", {})
        self._config.setdefault("labels", {})

        if os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
            self._config["app"]["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")
        if os.getenv("APP_TITLE"):
            self._config["app"]["title"] = os.getenv("APP_TITLE")
        if os.getenv("APP_DESCRIPTION"):
            self._config["app"]["description"] = os.getenv("APP_DESCRIPTION")

        if os.getenv("PAGE_ICON"):
            self._config["app"]["page_icon"] = os.getenv("PAGE_ICON")

    def _validate(self):
        if "title" not in self._config.get("app", {}):
            warnings.warn("Missing 'title' in [app] config")

    @property
    def app(self):
        return self._config.get("app", {})

    @property
    def ui(self):
        return self._config.get("ui", {})

    @property
    def sidebar(self):
        return self._config.get("sidebar", {})

    @property
    def styles(self):
        return self._config.get("styles", {})

    @property
    def labels(self):
        return self._config.get("labels", {})

    @property
    def prompts(self):
        return self._config.get("prompts", {})

    @property
    def messages(self):
        return self._config.get("messages", {})

    @property
    def chat(self):
        return self._config.get("chat", {})


# Global instance
config = Config()

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# Vector store settings
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Available embedding models from OpenRouter
AVAILABLE_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
