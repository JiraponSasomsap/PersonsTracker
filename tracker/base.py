from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.params = None
    
    @abstractmethod
    def update(self, boxes) -> 'BaseTrackerResults':
        pass

    def __call__(self, boxes) -> 'BaseTrackerResults':
        return self.update(boxes)

class BaseTrackerResults(ABC):
    @abstractmethod
    def boxes(self):
        pass