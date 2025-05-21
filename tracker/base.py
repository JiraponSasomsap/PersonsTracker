from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.detector = None
        self.params = None
    
    @abstractmethod
    def update(self, img):
        pass

class BaseTrackerResults(ABC):
    @abstractmethod
    def boxes(self):
        pass