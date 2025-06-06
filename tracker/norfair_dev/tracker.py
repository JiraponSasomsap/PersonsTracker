from norfair.tracker import (TrackedObject, 
                             Tracker, 
                             _TrackedObjectFactory, 
                             Detection)
import numpy as np
from .results import TrackedObjectResults
from norfair.filter import OptimizedKalmanFilterFactory

class tmng:
    '''Tracker Manager'''

    @staticmethod
    def conv2dataDict(**data):
        '''
        Convert column-based keyword arguments (dict of lists) to a list of row-based dicts.

        Example:
            conv2dataDict(x=[1,2], y=[3,4]) 
            => [{'x': 1, 'y': 3}, {'x': 2, 'y': 4}]

        Parameters:
            **data: Keyword arguments where each value is a list of the same length.

        Returns:
            List[dict]: List of dictionaries representing each row.

        Raises:
            ValueError: If input lists are not all the same length.
        '''
        keys = list(data.keys())
        vals = list(data.values())

        for i in range(1, len(vals)):
            if len(vals[0]) != len(vals[i]):
                raise ValueError("All values must have the same length")
            
        data_list = []

        for i in range(len(vals[0])):
            row = {k : data[k][i] for k in keys}
            data_list.append(row)
        
        return data_list

class CustomTrackedObject(TrackedObject):
    def __init__(self, 
                 obj_factory, 
                 initial_detection, 
                 hit_counter_max, 
                 initialization_delay, 
                 pointwise_hit_counter_max, 
                 detection_threshold, 
                 period, 
                 filter_factory, 
                 past_detections_length, 
                 reid_hit_counter_max, 
                 coord_transformations = None):
        super().__init__(obj_factory, 
                         initial_detection, 
                         hit_counter_max, 
                         initialization_delay, 
                         pointwise_hit_counter_max, 
                         detection_threshold, 
                         period, 
                         filter_factory, 
                         past_detections_length, 
                         reid_hit_counter_max, 
                         coord_transformations)

class CustomTracker(Tracker):
    def __init__(self,
                 distance_function, 
                 distance_threshold, 
                 hit_counter_max = 15, 
                 initialization_delay = None, 
                 pointwise_hit_counter_max = 4, 
                 detection_threshold = 0, 
                 filter_factory:'OptimizedKalmanFilterFactory' = None, 
                 past_detections_length = 4, 
                 reid_distance_function = None, 
                 reid_distance_threshold = 0, 
                 reid_hit_counter_max = None):
        
        if filter_factory is None:
            filter_factory = OptimizedKalmanFilterFactory()

        self.results_getter = TrackedObjectResults(self)

        super().__init__(distance_function, 
                         distance_threshold, 
                         hit_counter_max, 
                         initialization_delay, 
                         pointwise_hit_counter_max, 
                         detection_threshold, 
                         filter_factory, 
                         past_detections_length, 
                         reid_distance_function, 
                         reid_distance_threshold, 
                         reid_hit_counter_max)

    def set_tracker(self, custom_tracked_object = CustomTrackedObject, **kwds):
        '''
        Set custom tracked object factory and region configs.

        Parameters:
            custom_tracked_object: class to use for tracked objects
            **kwds: additional region configurations (e.g., roi, roni, roi1, roni1)

        Notes:
            - "roi" = Region Of Interest
            - "roni" = Region Of No Interest
        '''
        setattr(self, '_obj_factory', _TrackedObjectAutoFactory(custom_tracked_object))
        self.kwds = kwds
        return self

    def easy_update(
        self,
        points: np.ndarray,
        scores=None,
        data=None,
        label=None,
        embedding=None,
        **update_params
    ):
        '''
        Convenient wrapper to convert raw detection data into Detection objects 
        and update the tracker.

        Parameters:
            points (np.ndarray): An (N, 2) or (N, 4) array of detection points.
                - (x, y) for keypoints or center points
                - (x1, y1, x2, y2) for bounding boxes
            scores (list or array, optional): Confidence scores per detection.
            data (list, optional): Additional data for each detection.
            label (list, optional): Class labels per detection.
            embedding (list, optional): Feature vectors for Re-ID (Re-identification).
            **update_params: Extra parameters passed to the underlying `update()`.

        Returns:
            list: List of updated `TrackedObject` instances.
        
        Raises:
            TypeError: If `points` is not convertible to a numpy array.
            ValueError: If shape of `points` is invalid or if any extra param
                        (scores, data, label, embedding) has mismatched length.
        '''
        if not isinstance(points, np.ndarray):
            try:
                points = np.array(points)
            except Exception as e:
                raise TypeError("points must be convertible to a numpy.ndarray") from e

        if points.ndim != 2 or points.shape[1] not in (2, 4):
            raise ValueError("points must be a 2D array with shape (N, 2) or (N, 4)")

        param_dict = {
            'scores': scores,
            'data': data,
            'label': label,
            'embedding': embedding
        }

        for name, param in param_dict.items():
            if param is not None and len(param) != len(points):
                raise ValueError(f"Length mismatch: {name} has length {len(param)} but points has length {len(points)}")

        detections = []
        for i in range(len(points)):
            point = points[i]
            if points.shape[1] == 4:
                point = point.reshape(2, 2)
            detections.append(
                Detection(
                    points=point,
                    scores=scores[i] if scores is not None else None,
                    data=data[i] if data is not None else None,
                    label=label[i] if label is not None else None,
                    embedding=embedding[i] if embedding is not None else None
                )
            )

        return super().update(detections=detections, **update_params)
    
    @property
    def get(self):
        return self.results_getter

class _TrackedObjectAutoFactory(_TrackedObjectFactory):
    def __init__(self, object_class: type):
        super().__init__()
        if not issubclass(object_class, CustomTrackedObject):
            raise TypeError("object_class must be a subclass of TrackedObject")
        self.object_class = object_class

    def create(
        self,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory,
        past_detections_length: int,
        reid_hit_counter_max,
        coord_transformations,
    ) -> TrackedObject:
        return self.object_class(
            obj_factory=self,
            initial_detection=initial_detection,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=pointwise_hit_counter_max,
            detection_threshold=detection_threshold,
            period=period,
            filter_factory=filter_factory,
            past_detections_length=past_detections_length,
            reid_hit_counter_max=reid_hit_counter_max,
            coord_transformations=coord_transformations,
        )