# Renamed from types.py to avoid shadowing stdlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import json


@dataclass
class ModelResponse:
	"""
	Data structure for model responses.
	"""
	model: str
	prompt: str
	completion: str
	score: Optional[float] = None
	metadata: Optional[Dict[str, Any]] = None

	def to_dict(self) -> Dict[str, Any]:
		"""
		Convert ModelResponse to a dictionary, ensuring metadata is serializable.
		Returns:
			dict representation of the ModelResponse.
		"""
		d = asdict(self)
		if d.get("metadata") is None:
			d["metadata"] = {}
		return d

	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> "ModelResponse":
		"""
		Create a ModelResponse from a dictionary.
		Args:
			d: Dictionary with response fields.
		Returns:
			ModelResponse instance.
		"""
		return cls(
			model=d.get("model", ""),
			prompt=d.get("prompt", ""),
			completion=d.get("completion", ""),
			score=d.get("score"),
			metadata=d.get("metadata") or {},
		)


MetricInput = Dict[str, List[str]]  # Example: {"references": [...], "hypotheses": [...]}

__all__ = ["ModelResponse"]
