from typing import TYPE_CHECKING
import numpy as np
import cv2
from norfair.drawing.drawer import Drawer

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

    def active_last_detection_data(self, callback=None):
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
    
    def dd(self):
        print("[DEBUG] 'kwds' keys:", list(self.insts.kwds.keys()))
    
    @staticmethod
    def _set_roi_color_by_key(key):
        color = None
        if 'roi' in key:
            color = (0,255,0)
        elif 'roni' in key:
            color = (0,0,255)
        return color

    def draw_roi(self, 
                 img, 
                 roi=True, 
                 roni=True, 
                 font_size=1, 
                 font_thickness=2,
                 font_color=(255,255,255)):
        im = img.copy()
        overlay = img.copy()
        h, w = im.shape[:2]

        if roi or roni:
            for key, val in self.insts.kwds.items():
                color = self._set_roi_color_by_key(key)
                if color is None:
                    continue

                alpha_fill = 0.3

                np_val = np.array(val)

                abs_val = np_val * [w, h]

                abs_val = abs_val.astype(np.int32)

                cv2.fillPoly(overlay, [abs_val], color=color)
                cv2.polylines(im, [abs_val], isClosed=True, color=color, thickness=2)

                M = cv2.moments(abs_val)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = abs_val[0]

                Drawer.text(im, key, (cX, cY), font_size, font_color, font_thickness)

            cv2.addWeighted(overlay, alpha_fill, im, 1 - alpha_fill, 0, im)
        return im