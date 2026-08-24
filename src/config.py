"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Tải .env từ thư mục gốc của project (Lab/). Lab ưu tiên file
# cấu hình cục bộ để không vô tình dùng credential do shell/IDE inject.
_root = Path(__file__).parent.parent
_override_dotenv = os.getenv("DAY22_ENV_OVERRIDE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
load_dotenv(_root / ".env", override=_override_dotenv)

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
_tracing = (
    os.getenv("LANGSMITH_TRACING")
    or os.getenv("LANGCHAIN_TRACING_V2", "true")
)
_langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
_langsmith_project = (
    os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "day22-lab")
)
_langsmith_endpoint = (
    os.getenv("LANGSMITH_ENDPOINT")
    or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
)

# Export current and legacy names because supported SDK versions read both.
os.environ["LANGSMITH_TRACING"] = _tracing
os.environ["LANGCHAIN_TRACING_V2"] = _tracing
os.environ["LANGSMITH_API_KEY"] = _langsmith_key
os.environ["LANGCHAIN_API_KEY"] = _langsmith_key
os.environ["LANGSMITH_PROJECT"] = _langsmith_project
os.environ["LANGCHAIN_PROJECT"] = _langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = _langsmith_endpoint
os.environ["LANGCHAIN_ENDPOINT"] = _langsmith_endpoint

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = _langsmith_key
LANGSMITH_PROJECT = _langsmith_project
LANGSMITH_PROMPT_OWNER = os.getenv("LANGSMITH_PROMPT_OWNER", "").strip()


def _is_configured(value: str) -> bool:
    """Return whether an environment value is non-empty and not a template."""
    normalized = value.strip().lower()
    return bool(normalized) and not normalized.startswith("your_")


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []

    valid_providers = {"openai", "gemini", "anthropic", "ollama", "openrouter"}
    if PROVIDER not in valid_providers:
        print(
            f"⚠️  PROVIDER không hợp lệ: '{PROVIDER}'. "
            f"Chọn một trong: {', '.join(sorted(valid_providers))}."
        )
        return False

    if not _is_configured(LANGSMITH_API_KEY):
        missing.append("LANGSMITH_API_KEY hoặc LANGCHAIN_API_KEY")

    if PROVIDER == "openai" and not _is_configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not _is_configured(GOOGLE_API_KEY):
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not _is_configured(ANTHROPIC_API_KEY):
        missing.append("ANTHROPIC_API_KEY")
    elif PROVIDER == "openrouter" and not _is_configured(OPENROUTER_API_KEY):
        missing.append("OPENROUTER_API_KEY")
    if PROVIDER in {"anthropic", "openrouter"} and not _is_configured(
        OPENAI_API_KEY
    ):
        missing.append("OPENAI_API_KEY (dùng cho embeddings)")
    # Ollama: không cần API key

    if _tracing.strip().lower() not in {"1", "true", "yes", "on"}:
        missing.append("LANGSMITH_TRACING=true (hoặc LANGCHAIN_TRACING_V2=true)")

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  Provider: {PROVIDER.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    validate()
