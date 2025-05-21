from sklearn.metrics.pairwise import cosine_similarity
import cv2

def embedding_cosine_similarity(matched_not_init_trackers, unmatched_trackers):
    print('embedding_cosine_similarity')
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1
    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        cosine_sim = cosine_similarity([snd_embedding], [detection_fst.embedding])[0][0]
        if cosine_sim < 0.5:
            return cosine_sim
    return 1

def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    print('embedding_distance')
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(
            snd_embedding, detection_fst.embedding, cv2.HISTCMP_CORREL
        )
        if distance < 0.5:
            return distance
    return 1