from __future__ import annotations

import hashlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from torchvision.models import resnet50, ResNet50_Weights  # type: ignore
	_TORCHVISION_AVAILABLE = True
except Exception:
	_TORCHVISION_AVAILABLE = False


class InfectionModel:
	"""Wraps a pretrained CNN backbone with a small binary head.
	If torchvision is unavailable, falls back to a mock model with deterministic output from input hash.
	"""

	def __init__(self) -> None:
		self.model: nn.Module | None = None
		self.use_mock: bool = False

	def load(self) -> None:
		if _TORCHVISION_AVAILABLE:
			backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
			backbone.fc = nn.Identity()
			head = nn.Linear(2048, 2)
			self.model = nn.Sequential(backbone, head)
			self.model.eval()
		else:
			# Minimal linear just to satisfy interface; inference path uses mock
			self.model = nn.Linear(10, 2)
			self.use_mock = True

	@torch.inference_mode()
	def predict(self, input_tensor: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
		"""Returns (label, confidence, logits)."""
		if self.model is None:
			raise RuntimeError("Model not loaded")

		if self.use_mock:
			# Deterministic pseudo-probability from input hash
			data = input_tensor.detach().cpu().numpy().tobytes()
			h = int(hashlib.sha256(data).hexdigest(), 16)
			p_pos = (h % 1000) / 1000.0
			logits = torch.tensor([[1.0 - p_pos, p_pos]])
			probs = F.softmax(logits, dim=1)
		else:
			logits = self.model(input_tensor)
			probs = F.softmax(logits, dim=1)

		confidence, idx = torch.max(probs, dim=1)
		label = "Positive" if int(idx.item()) == 1 else "Negative"
		return label, float(confidence.item()), logits
