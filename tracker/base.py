from abc import ABC, abstractmethod
import norfair
import cv2

class BaseTracker(ABC):
    def __init__(self):
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

    def plot(self, img, alpha=1.0):
        h, w = img.shape[:2]

        boxes = self.boxes()
        ids = self.id()
        base_font_scale = 0.001 * (w + h) / 2
        font_scale = base_font_scale * alpha
        font_thickness = max(1, int(font_scale * 2))

        font = cv2.FONT_HERSHEY_SIMPLEX

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {obj_id}"
            org = (x1, y1 - 10)
            cv2.putText(img, label, org, font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
            cv2.putText(img, label, org, font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
        return img