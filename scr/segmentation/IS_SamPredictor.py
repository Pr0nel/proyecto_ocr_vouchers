import os
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# === CONFIGURACIÓN DEL USUARIO ===
ruta_imagen = "image_testOCR.jpg"          # Imagen a segmentar
modelo_sam = "vit_b"                        # "vit_b", "vit_l", o "vit_h"
#ruta_checkpoint = f"checkpoints/sam_vit_h_4b8939.pth"  # Ajusta si usas otro nombre
#ruta_checkpoint = f"checkpoints/sam_vit_l_0b3195.pth"
ruta_checkpoint = f"checkpoints/sam_vit_b_01ec64.pth" #La mejor para el caso de voucher
output_dir = "segmentos_sam"

os.makedirs(output_dir, exist_ok=True)

# === CARGAR MODELO SAM ===
print(f"Cargando modelo {modelo_sam}...")
sam = sam_model_registry[modelo_sam](checkpoint=ruta_checkpoint)
predictor = SamPredictor(sam)

# === CARGAR IMAGEN ===
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    raise FileNotFoundError(f"No se pudo leer la imagen en: {ruta_imagen}")
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# === PREDECIR MÁSCARAS ===
predictor.set_image(imagen_rgb)
masks, _, _ = predictor.predict()

# === EXTRAER Y GUARDAR SEGMENTOS ===
segmentos = []
for i, mask in enumerate(masks):
    mask_uint8 = mask.astype(np.uint8)
    if np.count_nonzero(mask_uint8) == 0:
        continue
    x, y, w, h = cv2.boundingRect(mask_uint8)
    if w * h < 100:  # evitar segmentos minúsculos
        continue
    segmento = imagen[y:y+h, x:x+w]
    if segmento.size > 0:
        segmentos.append((y, segmento))  # guardamos la posición vertical y el contenido

# === ORDENAR DE ARRIBA HACIA ABAJO Y GUARDAR ===
segmentos.sort(key=lambda seg: seg[0])  # orden por posición y (de arriba hacia abajo)
for idx, (_, seg) in enumerate(segmentos):
    salida_path = os.path.join(output_dir, f"{modelo_sam}_segmento_{idx}.png")
    cv2.imwrite(salida_path, seg)
    print(f"Guardado: {salida_path}")

print(f"\n✅ Proceso completado. {len(segmentos)} segmentos guardados en '{output_dir}'")
