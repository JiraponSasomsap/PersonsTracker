import norfair
import numpy as np

from .base import BaseTracker, BaseTrackerResults
from reid.osnet import osnet

class TrackerNorfair(BaseTracker):
    def __init__(self, **tracker_cfg):
        self.params = {'distance_function':'euclidean', 
                               'distance_threshold':50}
        self.params.update(tracker_cfg)
        self.tracker=norfair.Tracker(**self.params)

    def update(self, detections):
        norfair_detections = [norfair.Detection(points=points.reshape(2,2)) for points in detections]
        self.tracker.update(detections=norfair_detections)
        return TrackerNorfairResults(self)

class TrackerNorfairREID(TrackerNorfair):
    def __init__(self, reid:osnet, **tracker_cfg):
        super().__init__(**tracker_cfg)

class TrackerNorfairResults(BaseTrackerResults):
    def __init__(self, instance:TrackerNorfair):
        super().__init__()
        self.instance = instance
        self.is_process = False
        self._boxes = None
        self._id = None

    def _process_active_objects(self):
        active_obj = self.instance.tracker.get_active_objects()
        boxes = []
        ids = []
        for obj in active_obj:
            boxes.append(obj.last_detection.points)
            ids.append(obj.id)
        self._boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        self._id = np.array(ids, dtype=np.int32)

    def boxes(self):
        if not self.is_process:
            self._process_active_objects()
        return self._boxes
    
    def id(self):
        if not self.is_process:
            self._process_active_objects()
        return self._id