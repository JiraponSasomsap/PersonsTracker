from pathlib import Path
from typing import Union, Any
import numpy as np
from PIL import Image
import torch
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self):
        self.model_path:Path|str = None
        self.model = None
        self.kwargs:dict = None
        self.results = None

    @abstractmethod
    def set_predict_settings(self, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, 
                source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor],
                **kwargs) -> Any:
        pass
    
    def __call__(self, img):
        return self.predict(img)
    
class BaseDetectorResults(ABC):
    @abstractmethod
    def boxse(self):
        pass