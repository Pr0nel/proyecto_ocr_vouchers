import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import requests
"""
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

response = requests.get(url, verify=False, stream=True)

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open("checkpoints/sam_vit_h_4b8939.pth", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Checkpoint descargado en checkpoints/sam_vit_h_4b8939.pth")
"""

# Cargar el modelo SAM
#sam = sam_model_registry["default"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
#sam = sam_model_registry["vit_l"](checkpoint="checkpoints/sam_vit_l_0b3195.pth")
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

# Leer la imagen
imagen = cv2.imread("image_testOCR.jpg")
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Predecir las máscaras de segmentación
predictor.set_image(imagen_rgb)
masks, _, _ = predictor.predict()

# Guardar cada segmento como una imagen individual
for i, mask in enumerate(masks):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    segmento = imagen[y:y+h, x:x+w]
    cv2.imwrite(f"documento_b_{i}.png", segmento)