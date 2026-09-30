"""LLM client: Groq or Gemini, both via their OpenAI-compatible endpoints."""
import os

from openai import OpenAI

# Models are tried in order: when one hits its daily free-tier quota (or is retired),
# the agent falls back to the next. Each model has its own separate quota.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_models": [
            "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
            "gemini-3.5-flash", "gemini-3.5-flash-lite",
        ],
    },
}


def resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        if provider not in PROVIDERS:
            raise RuntimeError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got {provider!r}")
        return provider
    for name, cfg in PROVIDERS.items():
        if os.getenv(cfg["key_env"]):
            return name
    raise RuntimeError("No LLM key configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env")


def get_client() -> tuple[OpenAI, list[str], str]:
    """Return (client, models in fallback order, provider).

    LLM_MODEL may be a single model or a comma-separated fallback list.
    """
    provider = resolve_provider()
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['key_env']} is not set")
    env_models = [m.strip() for m in os.getenv("LLM_MODEL", "").split(",") if m.strip()]
    models = env_models or list(cfg["default_models"])
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"], max_retries=2, timeout=90)
    return client, models, provider
