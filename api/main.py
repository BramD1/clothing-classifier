import io
import asyncio
from pathlib import Path

import timm
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from pydantic import BaseModel
from timm.data import resolve_data_config, create_transform
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- App + rate limiter ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Clothing Classifier API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Accepted upload types ---
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}  # declared type (client-provided)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}                          # what PIL actually detects in the bytes

# --- (2) Model loader: runs ONCE at import/startup ---
CKPT_PATH = Path(__file__).parent / "best.pt"   # resolves next to this file, no matter where uvicorn starts
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)  # your own trusted file; set False if this errors on your torch version
model = timm.create_model(ckpt["model_name"], num_classes=len(ckpt["classes"]))
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()
classes = ckpt["classes"]
tf = create_transform(**resolve_data_config({}, model=model), is_training=False)

# --- Response schema (could move to schemas.py later) ---
class Prediction(BaseModel):
    label: str
    confidence: float

# --- (3a) Predictor: plain, synchronous ML logic (testable without the server) ---
def predict(image: Image.Image) -> Prediction:
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    i = int(probs.argmax().item())
    return Prediction(label=classes[i], confidence=round(probs[i].item(), 4))

# --- Health check (handy for Docker / deployment) ---
@app.get("/")
def health():
    return {"status": "ok", "classes": len(classes), "device": device}

# --- (3b) Route + (1) image loading from the upload ---
@app.post("/predict", response_model=Prediction)
@limiter.limit("10/minute")
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    # Layer 1: cheap early reject on the declared content type (can be spoofed)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,  # 415 Unsupported Media Type
            detail=f"Unsupported type '{file.content_type}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    # Layer 2: the real guarantee — the bytes must actually decode as an allowed image
    try:
        img = Image.open(io.BytesIO(await file.read()))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"File is a {img.format} image; allowed: {', '.join(sorted(ALLOWED_FORMATS))}.",
            )
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # offload CPU-bound inference so it doesn't block the event loop
    return await asyncio.to_thread(predict, img)