import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === CONFIGURACIÓN DEL USUARIO ===
ruta_imagen = "image_testOCR.jpg"          # Imagen a segmentar
modelo_sam = "vit_b"                        # "vit_b", "vit_l", o "vit_h"
#ruta_checkpoint = f"checkpoints/sam_vit_h_4b8939.pth"  # Ajusta si usas otro nombre
#ruta_checkpoint = f"checkpoints/sam_vit_l_0b3195.pth"
ruta_checkpoint = f"checkpoints/sam_vit_b_01ec64.pth"
output_dir = "segmentos_vouchers"
os.makedirs(output_dir, exist_ok=True)

# === CARGAR IMAGEN ===
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}")
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# === CARGAR MODELO SAM AUTOMÁTICO ===
print(f"Usando modelo SAM: {modelo_sam}")
sam = sam_model_registry[modelo_sam](checkpoint=ruta_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

# === GENERAR MÁSCARAS ===
print("Generando máscaras automáticas...")
masks = mask_generator.generate(imagen_rgb)

# === FILTRAR Y GUARDAR SEGMENTOS RECTANGULARES ===
print(f"Se encontraron {len(masks)} máscaras. Filtrando por forma/tamaño...")
segmentos_filtrados = []

for i, mask_data in enumerate(masks):
    mask = mask_data["segmentation"].astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)

    # Filtros: tamaño mínimo y proporción rectangular típica de vouchers
    area = w * h
    aspect_ratio = w / h if h > 0 else 0

    if not (0.5 < aspect_ratio < 3.0 and 15_000 < area < 300_000):
        continue  # ignorar regiones que no parecen vouchers

    segmento = imagen[y:y+h, x:x+w]
    if segmento.size > 0:
        segmentos_filtrados.append((y, segmento))  # guardar por orden vertical

# === ORDENAR Y GUARDAR ===
segmentos_filtrados.sort(key=lambda seg: seg[0])  # ordenar de arriba hacia abajo
for idx, (_, seg) in enumerate(segmentos_filtrados):
    ruta_salida = os.path.join(output_dir, f"{modelo_sam}_voucher_{idx}.png")
    cv2.imwrite(ruta_salida, seg)
    print(f"Guardado: {ruta_salida}")

print(f"\n✅ {len(segmentos_filtrados)} vouchers segmentados guardados en '{output_dir}'")
