import os
import cv2
import torch
import numpy as np
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === CONFIGURACIÓN DEL LOGGER ===
logging.basicConfig(
    level   =logging.INFO,
    format  ="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN DEL USUARIO ===
DIR_IMAGENES= "imagen_vouchers"
MODELO_SAM  = "vit_b"                        # "vit_b", "vit_l", o "vit_h"
#CHECKPOINT  = "checkpoints/sam_vit_h_4b8939.pth"  # Ajusta si usas otro nombre
#CHECKPOINT  = "checkpoints/sam_vit_l_0b3195.pth"
CHECKPOINT  = "checkpoints/sam_vit_b_01ec64.pth"
DIR_SALIDA  = "prueba_segmentos_vouchers"
os.makedirs(DIR_SALIDA, exist_ok=True)

# === CARGA DEL MODELO SAM ===
logger.info(f"Cargando modelo SAM: {MODELO_SAM}")
sam     = sam_model_registry[MODELO_SAM](checkpoint = CHECKPOINT)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device)

#mask_generator  = SamAutomaticMaskGenerator(sam)

mask_generator  = SamAutomaticMaskGenerator(
    model = sam,
    points_per_side         = 16, # Densidad de puntos usados para la segmentación automática.
    pred_iou_thresh         = 0.88, # Umbral de confianza de intersección (para descartar máscaras débiles).
    stability_score_thresh  = 0.95, # Umbral de estabilidad: rechaza máscaras que varían mucho frente a pequeñas perturbaciones.
    crop_n_layers           = 1, # Niveles de cropping jerárquico para segmentar en escalas.
    crop_n_points_downscale_factor = 2 # Reduce la densidad de puntos en niveles más pequeños.
    )

def limpiar_mascara(mask: np.ndarray, img_shape) -> np.ndarray:
    """Aplica operaciones morfológicas con kernel adaptativo"""
    kernel_size = (max(3, img_shape[0] // 50), max(3, img_shape[1] // 50))
    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    cleaned     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned     = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def procesar_imagen(ruta_img: str, nombre_img: str) -> int:
    imagen = cv2.imread(ruta_img)
    if imagen is None:
        logger.warning(f"No se pudo cargar: {ruta_img}")
        return 0
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    masks   = mask_generator.generate(img_rgb)

    segmentos = []
    for m in masks:
        mask= (m["segmentation"].astype(np.uint8)) * 255

        if cv2.countNonZero(mask) < 1000: # Omitir máscaras pequeñas
            continue
        
        mask= limpiar_mascara(mask, imagen.shape[:2])

        x, y, w, h = cv2.boundingRect(mask)

        area= w * h
        ar  = w / h if h > 0 else 0
        if not (0.5 < ar < 3.0 and 15000 < area < 300000):
            continue

        recorte = imagen[y:y+h, x:x+w]
        if recorte.size > 0:
            segmentos.append((y, recorte))

    # Ordena los recortes de arriba hacia abajo (eje y).
    # Así el índice voucher_0, voucher_1, etc., coincide con el orden visual de aparición en la imagen.
    segmentos.sort(key = lambda s: s[0])
    nombre_base = os.path.splitext(nombre_img)[0]
    for idx, (_, seg) in enumerate(segmentos):
        salida = os.path.join(DIR_SALIDA, f"{nombre_base}_voucher_{idx}.png")
        cv2.imwrite(salida, seg)
        logger.info(f"Guardado: {salida}")
    return len(segmentos)

# === PROCESAMIENTO MASIVO ===
extensiones = (".jpg", ".jpeg", ".png")
imagenes    = [f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(extensiones)]

total = 0
for nombre in imagenes:
    logger.info(f"Procesando '{nombre}'...")
    ruta    = os.path.join(DIR_IMAGENES, nombre)
    total  += procesar_imagen(ruta, nombre)

logger.info(f"Segmentación completa: {total} vouchers guardados en '{DIR_SALIDA}'")
