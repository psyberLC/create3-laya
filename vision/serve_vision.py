#!/usr/bin/env python3
"""Local open-vocabulary object detector for the robot's camera (OWLv2), served on 127.0.0.1.

Laya reads text only, so this turns camera frames into text-friendly detections:

  POST /v1/detect
  {"image": "<base64 JPEG or PNG>", "queries": ["a pallet", "a forklift"], "threshold": 0.2}
  -> {"detections": [{"query": "a pallet", "score": 0.41, "box": [x0, y0, x1, y1]}, ...],
      "width": 320, "height": 240, "ms": 48.1}

Run with the Laya venv (it has torch, transformers, fastapi and uvicorn):
  .venv-laya\\Scripts\\python.exe vision\\serve_vision.py
Model: google/owlv2-base-patch16-ensemble (Apache-2.0), downloaded from Hugging Face on first use.
"""

import argparse
import base64
import io
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = os.environ.get('VISION_MODEL', 'google/owlv2-base-patch16-ensemble')


class DetectRequest(BaseModel):
    image: str = Field(description='base64-encoded JPEG or PNG')
    queries: list[str] = Field(min_length=1, max_length=16)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)


def build_app(device):
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    dtype = torch.float16 if device == 'cuda' else torch.float32
    if dtype == torch.float16:
        model = model.half()
    app = FastAPI(title='create3-laya vision')

    @app.get('/v1/health')
    def health():
        return {'model': MODEL_ID, 'device': device}

    @app.post('/v1/detect')
    def detect(req: DetectRequest):
        start = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(base64.b64decode(req.image))).convert('RGB')
        except Exception as e:  # noqa: BLE001 - any decode failure is the client's input
            raise HTTPException(422, f'image could not be decoded: {e}')
        inputs = processor(text=[req.queries], images=image, return_tensors='pt').to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        # OWLv2 pads the image to a square before resizing; boxes come back in that frame.
        side = max(image.size)
        result = processor.post_process_grounded_object_detection(
            outputs, threshold=req.threshold, target_sizes=[(side, side)],
            text_labels=[req.queries])[0]
        detections = []
        for box, score, label in zip(result['boxes'].tolist(), result['scores'].tolist(),
                                     result['text_labels']):
            x0, y0, x1, y1 = box
            x0, x1 = max(0.0, x0), min(float(image.width), x1)
            y0, y1 = max(0.0, y0), min(float(image.height), y1)
            if x1 > x0 and y1 > y0:
                detections.append({'query': label, 'score': round(score, 4),
                                   'box': [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
        detections.sort(key=lambda d: d['score'], reverse=True)
        return {'detections': detections, 'width': image.width, 'height': image.height,
                'ms': round((time.perf_counter() - start) * 1000, 1)}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    uvicorn.run(build_app(args.device), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
