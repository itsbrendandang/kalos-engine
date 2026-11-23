"""Run with ``python -m kalos.providers.check`` to check local model setup."""
from __future__ import annotations

from kalos.assist.llm import model_info
from kalos.providers.ollama_provider import ollama_url
from kalos.providers.openai_compatible_provider import openai_url


def main() -> None:
    info = model_info(probe=True)
    provider = info["provider"]
    url = ollama_url() if provider == "ollama" else openai_url() if provider == "openai" else "n/a"
    print(f"Provider: {provider}")
    print(f"URL: {url}")
    print(f"Model: {info['model']}")
    print(f"Local: {info['local']}")
    print(f"Reachable: {info.get('reachable', 'not checked')}")
    print(f"Model pulled: {info.get('model_present', 'not checked')}")
    if provider == "ollama" and info.get("reachable") and not info.get("model_present"):
        print(f"Run: ollama pull {info['model']}")


if __name__ == "__main__":
    main()
