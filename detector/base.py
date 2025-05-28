from pathlib import Path
from typing import Union, Any
import numpy as np
from PIL import Image
import torch
from abc import ABC, abstractmethod
import cv2

class BaseDetector(ABC):
    def __init__(self):
        self.model_path:Path|str = None
        self.model = None
        self.kwargs:dict = None
        self.results = None
        self.original_img = None

    @abstractmethod
    def set_predict_settings(self, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, 
                source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor],
                **kwargs) -> 'BaseDetectorResults':
        pass
    
    def __call__(self, img):
        return self.predict(img)
    
    @property
    @abstractmethod
    def get(self) -> 'BaseDetectorResults':
        pass
    
class BaseDetectorResults(ABC):
    def __init__(self, instance:'BaseDetector'):
        super().__init__()
        self.instance = instance

    @abstractmethod
    def boxse(self, callback = None):
        pass

    def plot(self, img=None):
        if img is None:
            if self.original_img is None:
                raise ValueError
            plot = self.original_img.copy()
        else:
            plot = img.copy()
        for box in self.boxse():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(plot, (x1, y1), (x2, y2), (0, 0, 255), 5)
        return plot