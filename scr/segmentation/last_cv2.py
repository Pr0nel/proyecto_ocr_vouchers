import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

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
    # Convertimos la máscara a uint8
    mask_uint8 = mask.astype(np.uint8)

    # Chequear si la máscara tiene alguna región activa
    if np.count_nonzero(mask_uint8) == 0:
        print(f"Máscara {i} vacía, se omite.")
        continue  # Saltar a la siguiente máscara

    # Encontrar el bounding box
    x, y, w, h = cv2.boundingRect(mask_uint8)

    # Extraer el segmento
    segmento = imagen[y:y+h, x:x+w]

    # Verificar que el segmento no esté vacío
    if segmento.size == 0:
        print(f"Segmento {i} vacío después de recorte, se omite.")
        continue

    # Guardar la imagen del segmento
    cv2.imwrite(f"documento_last_b_{i}.png", segmento)
    print(f"Segmento {i} guardado exitosamente.")
