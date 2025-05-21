from .norfair_tracker import TrackerNorfair, TrackerNorfairResults
import norfair
from norfair.filter import OptimizedKalmanFilterFactory
import numpy as np
from .reid_distance_func import embedding_distance

class TrackerNorfairREID(TrackerNorfair):
    def __init__(self, 
                 distance_function='euclidean', 
                 distance_threshold = 50, 
                 hit_counter_max = 15, 
                 initialization_delay = None, 
                 pointwise_hit_counter_max = 4, 
                 detection_threshold = 0, 
                 filter_factory=OptimizedKalmanFilterFactory(), 
                 past_detections_length = 4, 
                 reid_distance_function=embedding_distance, 
                 reid_distance_threshold = 0.5, 
                 reid_hit_counter_max = 500):
        super().__init__(distance_function, distance_threshold, hit_counter_max, initialization_delay, pointwise_hit_counter_max, detection_threshold, filter_factory, past_detections_length, reid_distance_function, reid_distance_threshold, reid_hit_counter_max)

    def update(self, detections, embeddings=None):
        if embeddings is not None:
            assert len(detections) == len(embeddings)
        else:
            embeddings = [None] * len(detections)
        norfair_detections = [
            norfair.Detection(
                points=points.reshape(2,2), 
                embedding=embed
            ) for points, embed in zip(detections, embeddings)
        ]
        self.tracker.update(detections=norfair_detections)
        return TrackerNorfairResultsREID(self)

class TrackerNorfairResultsREID(TrackerNorfairResults):
    def __init__(self, instance):
        super().__init__(instance)

    def _process_active_objects(self):
        active_obj = self.instance.tracker.get_active_objects()
        boxes = []
        ids = []
        embedding = []
        for obj in active_obj:
            boxes.append(obj.last_detection.points)
            ids.append(obj.id)
            embedding.append(obj.last_detection.embedding)
        self._boxes = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        self._id = np.array(ids, dtype=np.int32)
        self._embedding = embedding
    
    def boxes(self):
        return super().boxes()
    
    def id(self):
        return super().id()

    def embedding(self):
        return self._embedding
