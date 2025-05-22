import sys
sys.path.append('..')


if __name__ == '__main__':
    import cv2
    import detector
    import tracker

yolo = detector.DetectorYOLO(r'../../../yolov8n-pose.pt')
yolo.set_predict_settings(verbose=False)
tracker = tracker.TrackerNorfair()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = yolo(frame)
    active_obj = tracker.update(detections=detections.boxse())
    print(f'id : {active_obj.id()}', f'boxes : {active_obj.boxes()}')
cap.release()