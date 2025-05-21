from abc import ABC, abstractmethod

class BaseREID(ABC):
    def __init__(self):
        self.params = {}
        self.extractor = None
    
    @abstractmethod
    def extract_feature_imfile(self, img_path):
        pass
    
    @abstractmethod
    def extract_feature(self, img_cv2):
        pass
    