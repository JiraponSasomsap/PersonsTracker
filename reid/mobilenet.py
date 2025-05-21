# Not yet supported
raise NotImplementedError('NotImplementedError')

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# โหลด model + transform
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).features.eval()
transform = weights.transforms()

def extract_feature(image):
    image = image.convert("RGB")  # 🔧 แปลงเป็น RGB
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = model(img)
    return feat.flatten().numpy()

vec1 = extract_feature(Image.open("image1.png"))
vec2 = extract_feature(Image.open("image2.png"))
sim = cosine_similarity([vec1], [vec2])[0][0]
print(f"Cosine similarity: {sim:.4f}")