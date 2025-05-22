from abc import ABC, abstractmethod
import norfair

class BaseTracker(ABC):
    def __init__(self, img):
        super().__init__()
        self.tracker:norfair.Tracker = None
        self.params:dict = None
    
    @abstractmethod
    def update(self, boxes) -> 'BaseTrackerResults':
        pass

    def __call__(self, boxes) -> 'BaseTrackerResults':
        return self.update(boxes)
    
    @property
    @abstractmethod
    def get(self) -> 'BaseTrackerResults':
        pass

class BaseTrackerResults(ABC):
    def __init__(self, instance:BaseTracker):
        super().__init__()
        self.instance = instance

    @abstractmethod
    def boxes(self):
        pass

    @abstractmethod
    def id(self):
        pass