"""全局配置，从 .env 文件读取。"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载工作目录下的 .env
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("未找到 OPENAI_API_KEY，请在 .env 文件中配置。")
    return key


def get_base_url() -> str | None:
    return os.getenv("OPENAI_BASE_URL") or None


def get_default_model() -> str:
    return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")


def get_memory_dir() -> Path:
    directory = os.getenv("SHORTCHAIN_MEMORY_DIR", ".shortchain_memory")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
