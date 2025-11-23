from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from .utils import preprocess_image, overlay_heatmap_on_image
from .model import InfectionModel
from .gradcam import GradCAM


app = FastAPI(title="Brain Amoebic Infection Detection System", version="0.1.0")

# Allow local dev frontends
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Lazy global singletons
_model: InfectionModel | None = None
_cam: GradCAM | None = None


def get_model() -> InfectionModel:
	global _model
	if _model is None:
		_model = InfectionModel()
		_model.load()
	return _model


def get_cam() -> GradCAM:
	global _cam
	if _cam is None:
		m = get_model().model
		_cam = GradCAM(model=m, target_layer_name="layer4") if m is not None else GradCAM(model=get_model().model, target_layer_name="layer4")
	return _cam


@app.get("/health")
async def health() -> Dict[str, Any]:
	return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
	try:
		contents = await file.read()
		image = Image.open(io.BytesIO(contents)).convert("RGB")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

	model = get_model()
	cam = get_cam()

	# Preprocess for model
	input_tensor = preprocess_image(image)

	# Run prediction
	prediction_label, confidence, _ = model.predict(input_tensor)

	# Grad-CAM heatmap
	heatmap = cam.generate_cam(input_tensor, target_class=1 if prediction_label == "Positive" else 0)
	overlay = overlay_heatmap_on_image(image, heatmap=heatmap, alpha=0.45)

	# Encode overlay as base64 PNG
	buf = io.BytesIO()
	overlay.save(buf, format="PNG")
	buf.seek(0)
	gradcam_b64 = base64.b64encode(buf.read()).decode("utf-8")

	# Simple textual summary
	summary = (
		"Infection detected with high activation in cortical regions."
		if prediction_label == "Positive"
		else "No abnormal region detected."
	)

	return {
		"prediction": prediction_label,
		"confidence": confidence,
		"gradcam": gradcam_b64,
		"summary": summary,
	}


@app.post("/report")
async def report(payload: Dict[str, Any]) -> Response:
	"""
	Generate a PDF report from frontend-provided data.
	Expected payload keys: prediction (str), confidence (float), timestamp (str|optional),
	optional base64 images: original (str), gradcam (str)
	"""
	from reportlab.lib.pagesizes import A4
	from reportlab.pdfgen import canvas
	from reportlab.lib.utils import ImageReader

	prediction = payload.get("prediction", "Unknown")
	confidence = float(payload.get("confidence", 0.0))
	timestamp = payload.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
	original_b64 = payload.get("original")
	gradcam_b64 = payload.get("gradcam")
	summary = payload.get("summary", "")

	buf = io.BytesIO()
	c = canvas.Canvas(buf, pagesize=A4)
	width, height = A4

	# Header
	c.setFont("Helvetica-Bold", 18)
	c.drawString(40, height - 50, "Brain Amoebic Infection Report")
	c.setFont("Helvetica", 11)
	c.drawString(40, height - 70, f"Timestamp: {timestamp}")

	# Result
	c.setFont("Helvetica-Bold", 12)
	c.drawString(40, height - 110, f"Prediction: {prediction}")
	c.setFont("Helvetica", 12)
	c.drawString(40, height - 130, f"Confidence: {confidence * 100:.1f}%")
	if summary:
		c.drawString(40, height - 150, f"Summary: {summary}")

	# Images
	y_cursor = height - 200
	def draw_b64(img_b64: str, x: float, y: float, max_w: float = 240, max_h: float = 240):
		try:
			data = base64.b64decode(img_b64)
			img = Image.open(io.BytesIO(data)).convert("RGB")
			w, h = img.size
			scale = min(max_w / w, max_h / h)
			new_w, new_h = int(w * scale), int(h * scale)
			c.drawImage(ImageReader(img), x, y, width=new_w, height=new_h)
		except Exception:
			pass

	if original_b64:
		c.setFont("Helvetica-Bold", 11)
		c.drawString(40, y_cursor + 250, "Original Image")
		draw_b64(original_b64, 40, y_cursor)

	if gradcam_b64:
		c.setFont("Helvetica-Bold", 11)
		c.drawString(320, y_cursor + 250, "Grad-CAM Overlay")
		draw_b64(gradcam_b64, 320, y_cursor)

	c.showPage()
	c.save()
	buf.seek(0)
	pdf_bytes = buf.read()
	return Response(content=pdf_bytes, media_type="application/pdf", headers={
		"Content-Disposition": "attachment; filename=brain_amoebic_report.pdf"
	})


if __name__ == "__main__":
	# For local debugging: uvicorn backend.app.main:app --reload --port 8000
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
