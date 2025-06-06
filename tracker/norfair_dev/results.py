from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import CustomTracker

class TrackedObjectResults:
    def __init__(self, insts:"CustomTracker"):
        self.insts = insts

    def active_id(self, callback=None):
        oo = [
            o.id
            for o in self.insts.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]
        if callback is not None:
            return callable(oo)
        return oo

    def active_age(self, callback=None):
        oo = [
            o.age
            for o in self.insts.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]
        if callback is not None:
            return callable(oo)
        return oo

    def active_label(self, callback=None):
        oo = [
            o.label
            for o in self.insts.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]
        if callback is not None:
            return callable(oo)
        return oo

    def active_last_detection_data(self, callback):
        oo = [
            o.last_detection.data
            for o in self.insts.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]
        if callback is not None:
            return callable(oo)
        return oo

    def active_last_detection_points(self, callback=None):
        oo = [
            o.last_detection.points
            for o in self.insts.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]
        if callback is not None:
            return callable(oo)
        return oo