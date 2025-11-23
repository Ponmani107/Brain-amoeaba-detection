from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def preprocess_image(image: Image.Image) -> torch.Tensor:
	"""Resize to 224x224 and normalize for ImageNet models. Returns tensor shape (1,3,224,224)."""
	image_resized = image.resize((224, 224))
	arr = np.asarray(image_resized).astype(np.float32) / 255.0
	mean = np.array(IMAGENET_MEAN, dtype=np.float32)
	std = np.array(IMAGENET_STD, dtype=np.float32)
	arr = (arr - mean) / std
	arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # HWC -> CHW, ensure float32
	tensor = torch.from_numpy(arr).unsqueeze(0).float()
	return tensor


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
	# Accept (H,W), (N,H,W), (N,1,H,W)
	if hm.ndim == 2:
		pass
	elif hm.ndim == 3:
		# (N,H,W) -> take first sample
		hm = hm[0]
	elif hm.ndim == 4:
		# (N,1,H,W) -> first sample, drop channel
		hm = hm[0, 0]
	else:
		raise ValueError("Unsupported heatmap shape")
	# Normalize 0..1
	hm = hm.astype(np.float32)
	mn, mx = float(hm.min()), float(hm.max())
	if mx - mn < 1e-12:
		return np.zeros_like(hm, dtype=np.float32)
	return (hm - mn) / (mx - mn)


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
	"""Overlay a heatmap (0..1) onto image and return an RGB PIL image."""
	try:
		import cv2  # type: ignore
	except Exception:
		# Fallback: simple red mask if OpenCV missing
		img = image.convert("RGB")
		hm = _normalize_heatmap(heatmap)
		w, h = img.size
		hm_resized = torch.nn.functional.interpolate(
			torch.from_numpy(hm).unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
		).squeeze().numpy()
		overlay = np.array(img).astype(np.float32)
		overlay[:, :, 0] = np.clip(overlay[:, :, 0] * (1 - alpha) + 255 * alpha * hm_resized, 0, 255)
		return Image.fromarray(overlay.astype(np.uint8))

	img = image.convert("RGB")
	w, h = img.size
	hm = _normalize_heatmap(heatmap)

	# Resize heatmap to image size
	hm_resized = torch.nn.functional.interpolate(
		torch.from_numpy(hm).unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
	).squeeze().numpy()

	hm_uint8 = (hm_resized * 255).astype(np.uint8)
	hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
	overlay = (1 - alpha) * np.array(img).astype(np.float32) + alpha * hm_color.astype(np.float32)
	overlay = np.clip(overlay, 0, 255).astype(np.uint8)
	return Image.fromarray(overlay)
