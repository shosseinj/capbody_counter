"""Persistent local CountGD++ API using the existing Python environment."""
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from local_count import Counter, annotate, encode_image
from cap_body import predict_cap_body

counter = None
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global counter
    torch.set_num_threads(int(os.environ.get("COUNT_THREADS", "4")))
    counter = Counter()
    yield
    counter = None


app = FastAPI(title="CapBody Counter", version="1.0", lifespan=lifespan,
              description="Official CountGD++ weights. Upload an image, name the object category, "
                          "and optionally provide positive/negative example boxes in image pixels after EXIF rotation. "
                          "Predictions may include partly visible objects; fully hidden inventory is not estimated.")


WEB = Path(__file__).resolve().parent / 'web'
app.mount('/assets', StaticFiles(directory=WEB), name='assets')


@app.get('/', include_in_schema=False)
def workspace():
    return FileResponse(WEB / 'index.html')


@app.get('/api/info')
def index():
    return {"model": "CountGD++", "docs": "/docs", "health": "/health", "endpoint": "POST /count"}


def read_image(file):
    payload = file.file.read(15*1024**2 + 1)
    if len(payload) > 15*1024**2:
        raise HTTPException(413, 'Maximum upload size is 15 MiB')
    with Image.open(io.BytesIO(payload)) as opened:
        if opened.width * opened.height > 25_000_000:
            raise HTTPException(413, 'Maximum image size is 25 megapixels')
        return ImageOps.exif_transpose(opened).convert('RGB')


@app.get("/health")
def health():
    return {"status": "ok" if counter is not None else "loading", "model": "CountGD++",
            "ready": counter is not None, "device": counter.device if counter is not None else None,
            "parameters": counter.parameter_count if counter is not None else None,
            "source": counter.source if counter is not None else None}


@app.post("/count")
def count(file: UploadFile = File(...), prompt: str = Form("bottle", max_length=500),
          threshold: float = Form(0.23, gt=0, lt=1), exemplars: str = Form("[]", max_length=10000),
          negative_texts: str = Form("[]", max_length=10000),
          negative_exemplars: str = Form("null", max_length=20000), include_image: bool = Form(True)):
    if counter is None:
        raise HTTPException(503, "Model is loading")
    try:
        image = read_image(file)
        # Model predictor access is serialized to bound peak memory.
        with lock:
            result = counter.predict(image, text=prompt, threshold=threshold,
                                     exemplars=json.loads(exemplars), negative_texts=json.loads(negative_texts),
                                     negative_exemplars=json.loads(negative_exemplars))
        if include_image:
            result["annotated_image_base64"] = encode_image(annotate(image, result))
        return result
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, f"Inference failed: {exc}") from exc


@app.post('/count/cap-body')
def count_cap_body(file: UploadFile = File(...), cap_exemplars: str = Form(..., max_length=3000),
                   body_exemplars: str = Form('[]', max_length=3000),
                   body_threshold: float = Form(.23, gt=0, lt=1),
                   cap_threshold: float = Form(.23, gt=0, lt=1)):
    if counter is None:
        raise HTTPException(503, 'Model is loading')
    try:
        image = read_image(file)
        cap_boxes, body_boxes = json.loads(cap_exemplars), json.loads(body_exemplars)
        with lock:
            return predict_cap_body(counter, image, cap_boxes, body_boxes, body_threshold, cap_threshold)
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, f'Inference failed: {exc}') from exc
