from ultralytics import YOLO
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import torch

from .base import BaseDetector, BaseDetectorResults

class DetectorYOLO(BaseDetector):
    def __init__(self, 
                 model: str | Path, 
                 task=None, 
                 verbose=False, 
                 **kwargs):
        super().__init__()
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model_path = model_path
            self.model = YOLO(model=model_path, task=task, verbose=verbose)
        else:
            raise TypeError(f"Invalid model type: {type(model)}. Expected str or Path.")
        self.kwargs = {}
        self.set_predict_settings(device='cpu')
        self.set_predict_settings(**kwargs)

    def set_predict_settings(self, **kwargs):
        self.kwargs.update(kwargs)

    def predict(self,
                source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor],
                **kwargs):
        _kwargs = self.kwargs.copy()
        _kwargs.update(kwargs)
        self.results = self.model.predict(source=source, **_kwargs)[0]
        return GetResults(self)
    
    def __call__(self, img):
        return self.predict(img, **self.kwargs)
        
class GetResults(BaseDetectorResults):
    def __init__(self, instance:DetectorYOLO):
        super().__init__()
        self.instance = instance
    
    def boxse(self):
        results = self.instance.results
        if results is None:
            raise RuntimeError("No prediction results found. Please run `predict()` first.")
        
        boxes = results.boxes
        if boxes is not None and boxes.xyxy is not None:
            return boxes.xyxy.cpu().numpy()
        else:
            return np.empty((0, 4), dtype=np.float32)
        
    def imcrops(self):
        results = self.instance.results
        img = results.orig_img
        h, w = img.shape[:2]
        cropped = []

        for box in self.boxse():
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cropped.append(img[y1:y2, x1:x2])

        return cropped