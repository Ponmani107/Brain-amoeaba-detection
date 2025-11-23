from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
	"""Minimal Grad-CAM for CNN backbones (e.g., ResNet).
	Assumes the target layer produces feature maps of shape (N, C, H, W).
	"""

	def __init__(self, model: nn.Module, target_layer_name: str = "layer4") -> None:
		self.model = model
		self.model.eval()
		self.target_layer_name = target_layer_name
		self.activations: Optional[torch.Tensor] = None
		self.gradients: Optional[torch.Tensor] = None

		target_layer = self._find_layer(self.model, target_layer_name)
		if target_layer is None:
			raise ValueError(f"Could not find layer '{target_layer_name}' in model")

		def forward_hook(_module, _input, output):
			self.activations = output.detach()

		def backward_hook(_module, grad_input, grad_output):
			self.gradients = grad_output[0].detach()

		target_layer.register_forward_hook(forward_hook)
		target_layer.register_full_backward_hook(backward_hook)

	def _find_layer(self, module: nn.Module, name: str) -> Optional[nn.Module]:
		for n, m in module.named_modules():
			if n.endswith(name):
				return m
		return None

	def generate_cam(self, x: torch.Tensor, target_class: int) -> np.ndarray:
		"""Returns heatmap array in shape (N, H, W) normalized to 0..1."""
		self.model.zero_grad(set_to_none=True)
		logits = self.model(x)
		if logits.ndim != 2:
			raise RuntimeError("Expected logits shape (N, C)")

		score = logits[:, target_class]
		score.sum().backward()

		if self.activations is None or self.gradients is None:
			# Fallback uniform heatmap to keep UX consistent when hooks unavailable
			with torch.no_grad():
				return torch.ones((x.shape[0], 7, 7)).numpy()

		# Global average pooling over gradients
		weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (N,C,1,1)
		weighted = (weights * self.activations).sum(dim=1, keepdim=True)  # (N,1,H,W)
		heatmap = torch.relu(weighted)
		# Normalize per-sample
		heatmap_min = heatmap.amin(dim=(2, 3), keepdim=True)
		heatmap_max = heatmap.amax(dim=(2, 3), keepdim=True)
		heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
		# Return (N,H,W)
		heatmap = heatmap.squeeze(1)
		return heatmap.detach().cpu().numpy()
