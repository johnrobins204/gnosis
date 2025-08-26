from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..types import ModelResponse


class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """
    def __init__(self, identifier: str, config: Optional[Dict[str, Any]] = None, api_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            identifier: Model identifier string.
            config: Optional configuration dict.
            api_params: Optional API parameters dict.
        """
        self.identifier = identifier
        self.config = config or {}
        self.api_params = api_params or {}

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a completion for the prompt and return a ModelResponse.
        Args:
            prompt: Input prompt string.
            **kwargs: Additional generation parameters.
        Returns:
            ModelResponse object.
        """
        raise NotImplementedError


class GoogleModel(LanguageModel):
    """
    Stub implementation of a Google-backed language model.
    """
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a completion using GoogleModel (stub).
        """
        # Minimal stub implementation: echo prompt. Replace with real API call later.
        return ModelResponse(
            model=f"google:{self.identifier}",
            prompt=prompt,
            completion=f"echo:{prompt}",
            score=None,
            metadata={"provider": "google", "api_params": self.api_params},
        )


class OllamaModel(LanguageModel):
    """
    Stub implementation of an Ollama-backed language model.
    """
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a completion using OllamaModel (stub).
        """
        # Minimal stub implementation: echo with provider tag.
        return ModelResponse(
            model=f"ollama:{self.identifier}",
            prompt=prompt,
            completion=f"ollama_echo:{prompt}",
            score=None,
            metadata={"provider": "ollama", "api_params": self.api_params},
        )


def get_model_instance(identifier: str, config: Optional[Dict[str, Any]] = None, api_params: Optional[Dict[str, Any]] = None) -> LanguageModel:
    """
    Factory returning a LanguageModel instance.
    Identifier prefixes supported: 'google:<name>' and 'ollama:<name>'.
    Default provider is GoogleModel when no prefix is present.
    Supports 'experiment:<name>[:<model_id>]' via adapter.
    """
    """
    Factory returning a LanguageModel instance.
    Identifier prefixes supported: 'google:<name>' and 'ollama:<name>'.
    Default provider is GoogleModel when no prefix is present.
    Supports 'experiment:<name>[:<model_id>]' via adapter.
    """
    # new experiment adapter handling
    if identifier.startswith("experiment:"):
        # lazy import adapter to avoid import-time deps
        from .adapters import experiment_adapter  # type: ignore
        return experiment_adapter.get_experiment_model_instance(identifier, config=config, api_params=api_params)

    if identifier.startswith("google:"):
        name = identifier.split(":", 1)[1]
        return GoogleModel(name, config=config, api_params=api_params)
    if identifier.startswith("ollama:"):
        name = identifier.split(":", 1)[1]
        return OllamaModel(name, config=config, api_params=api_params)
    # fallback
    return GoogleModel(identifier, config=config, api_params=api_params)