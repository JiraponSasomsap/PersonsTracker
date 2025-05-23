import cv2
from pathlib import Path

def version():
    v = Path(__file__).parents[1] / "VERSION"
    print(v)
    return v.read_text().strip()

def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()