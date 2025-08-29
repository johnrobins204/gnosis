
import requests
from typing import List, Dict, Any, Optional

def list_google_models() -> List[Dict[str, Any]]:
    """
    List available Google models (static list, as there is no public API for model listing).
    Returns a list of model metadata dicts.
    """
    # Update this list as Google releases new models or APIs
    return [
        {"provider": "google", "id": "gemini-pro", "name": "Gemini Pro"},
        {"provider": "google", "id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        {"provider": "google", "id": "palm-2", "name": "PaLM 2"},
    ]

def list_local_models(server: str = "ollama", **kwargs) -> List[Dict[str, Any]]:
    """
    Discover local models from supported servers. Currently supports 'ollama'.
    Returns a list of model metadata dicts.
    Args:
        server: Which local server to query (default: 'ollama')
        kwargs: Server-specific options (e.g., base_url)
    """
    if server == "ollama":
        return list_ollama_models(base_url=kwargs.get("base_url", "http://localhost:11434"))
    # Add more elif blocks for other servers (e.g., 'lmstudio', 'vllm')
    raise NotImplementedError(f"Local model server '{server}' not supported yet.")


# --- Remote Model Provider Discovery ---
import os

def list_openai_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available OpenAI models using the OpenAI API.
    Returns a list of model metadata dicts, or an error if API key is missing/invalid.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [{"provider": "openai", "error": "Missing OpenAI API key"}]
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI returns {"data": [{"id": ..., ...}, ...]}
        return [
            {"provider": "openai", "id": m.get("id"), **m}
            for m in data.get("data", [])
        ]
    except Exception as e:
        return [{"provider": "openai", "error": str(e)}]

def list_anthropic_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available Anthropic models using the Anthropic API.
    Returns a list of model metadata dicts, or an error if API key is missing/invalid.
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return [{"provider": "anthropic", "error": "Missing Anthropic API key"}]
    try:
        # Anthropic's model listing endpoint is not public; use a static list or update as needed
        # Example static list (Claude models)
        return [
            {"provider": "anthropic", "id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
            {"provider": "anthropic", "id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
            {"provider": "anthropic", "id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        ]
    except Exception as e:
        return [{"provider": "anthropic", "error": str(e)}]

def list_remote_models(providers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    List available models from remote providers (OpenAI, Google, etc.).
    Returns a unified list of model metadata dicts.
    Args:
        providers: List of provider names to query (default: all supported)
    """
    results = []
    providers = providers or ["openai", "google"]
    if "openai" in providers:
        results.extend(list_openai_models())
    if "google" in providers:
        results.extend(list_google_models())
    return results

def list_ollama_models(base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """
    Query Ollama REST API for available models.
    Returns a list of model metadata dicts.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"models": [{"name": ..., "size": ..., ...}, ...]}
        models = data.get("models", [])
        for m in models:
            m["provider"] = "ollama"
            m["server_name"] = "ollama"
        return models
    except Exception as e:
        return [{"error": str(e)}]
