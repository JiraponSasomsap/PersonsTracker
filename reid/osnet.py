import faiss
import cv2
from torchreid.utils import FeatureExtractor
from PIL import Image
from .base import BaseREID

class osnet(BaseREID):
    def __init__(self, model_name='osnet_x1_0', device='cpu'):
        self.params = {'model_name':model_name, 'device':device}
        self.extractor = FeatureExtractor(**self.params)

    def extract_feature_imfile(self, img_path):
        img = Image.open(img_path).convert('RGB')
        feat = self.extractor(img)
        return feat[0].numpy().astype('float32') 
    
    def extract_feature(self, img_cv2):
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  
        img_pil = Image.fromarray(img_rgb)
        feat = self.extractor(img_pil)
        return feat[0].numpy().astype('float32')

    def extract_feature_batch(self, img_cv2_list):
        img_np_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_cv2_list]
        feats = self.extractor(img_np_list)
        return feats

    # def __call__(self, query_feature, features):
    #     d = features.shape[1]
    #     faiss.normalize_L2(features)
    #     index = faiss.IndexFlatIP(d)  # inner product search == cosine similarity เมื่อ normalize แล้ว
    #     index.add(features)
    #     print(f"Indexed {index.ntotal} feature vectors.")
    #     faiss.normalize_L2(query_feature.reshape(1, -1))
    #     k = 3  # หา top-3 ใกล้เคียงที่สุด
    #     distances, indices = index.search(query_feature.reshape(1, -1), k)        
    #     print(f"Top {k} similar images:")
    #     for rank, idx in enumerate(indices[0]):
    #         print(f"Rank {rank+1}: {image_paths[idx]}, similarity score: {distances[0][rank]:.4f}")