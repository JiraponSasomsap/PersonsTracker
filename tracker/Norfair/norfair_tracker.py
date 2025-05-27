import norfair
import numpy as np
from ..base import BaseTracker, BaseTrackerResults
from norfair.filter import OptimizedKalmanFilterFactory

class TrackerNorfair(BaseTracker):
    def __init__(self, 
                 distance_function='euclidean',
                 distance_threshold: float=50,
                 hit_counter_max: int = 15,
                 initialization_delay: int | None = None,
                 pointwise_hit_counter_max: int = 4,
                 detection_threshold: float = 0,
                 filter_factory = OptimizedKalmanFilterFactory(),
                 past_detections_length: int = 4,
                 reid_distance_function = None,
                 reid_distance_threshold: float = 0,
                 reid_hit_counter_max: int | None = None):
        super().__init__()
        self.params = {
            'distance_function':distance_function,
            'distance_threshold':distance_threshold,
            'hit_counter_max':hit_counter_max,
            'initialization_delay':initialization_delay,
            'pointwise_hit_counter_max':pointwise_hit_counter_max,
            'detection_threshold':detection_threshold,
            'filter_factory':filter_factory,
            'past_detections_length':past_detections_length,
            'reid_distance_function':reid_distance_function,
            'reid_distance_threshold':reid_distance_threshold,
            'reid_hit_counter_max':reid_hit_counter_max,
        }
        self.tracker=norfair.Tracker(**self.params)

    def update(self, detections):
        norfair_detections = [norfair.Detection(points=points.reshape(2,2)) for points in detections]
        self.tracker.update(detections=norfair_detections)
        return TrackerNorfairResults(self)
    
    @property
    def get(self):
        return TrackerNorfairResults(self)

class TrackerNorfairResults(BaseTrackerResults):
    def __init__(self, instance:'TrackerNorfair'):
        super().__init__(self)
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