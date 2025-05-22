import sys
sys.path.append('..')

if __name__ == '__main__':

    import cv2
    import detector as Detector
    import tracker as Tracker
    import reid as REID
    from utils.utils import get_hist

yolo = Detector.DetectorYOLO(r'../../../yolov8n-pose.pt')
yolo.set_predict_settings(verbose=False)
tracker = Tracker.TrackerNorfairREID(
    # reid_distance_function=Tracker.reid_distance_func.embedding_cosine_similarity
)
reid = REID.osnet()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = yolo(frame)
    active_obj = tracker.update(
        detections=detections.boxse(), 
        embeddings=[get_hist(crop) for crop in detections.imcrops()]
        # embeddings=reid.extract_feature_batch(detections.imcrops())
    )
    print(f'id : {active_obj.id()}', f'boxes : {active_obj.boxes()}')

cap.release()