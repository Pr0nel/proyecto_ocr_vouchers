#scr/segmentation/voucher_segmentation.py
from pathlib import Path
from typing import List

import cv2
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def limpiar_mascara(mask: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Aplica operaciones morfológicas con kernel adaptativo para limpiar ruido."""
    kernel_size = (max(3, img_shape[0] // 50), max(3, img_shape[1] // 50))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

class Segmenter:
    def __init__(self, model_name: str, checkpoint: str, device: torch.device):
        self.device = device
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
        self.masks_cache = {}

    def segment(self, image_path: Path) -> List[Image.Image]:
        # Generar o recuperar máscaras
        if image_path in self.masks_cache:
            masks = self.masks_cache[image_path]
        else:
            img = Image.open(image_path).convert("RGB")
            masks = self.mask_generator.generate(np.array(img))
            self.masks_cache[image_path] = masks

        img = Image.open(image_path).convert("RGB")
        segments = []
        img_h, img_w = img.size[1], img.size[0]

        for m in masks:
            raw = (m['segmentation'].astype(np.uint8)) * 255
            # Filtrar por tamaño mínimo
            if cv2.countNonZero(raw) < 1000:
                continue

            # Limpiar máscara
            clean = limpiar_mascara(raw, (img_h, img_w))
            x, y, w, h = cv2.boundingRect(clean)

            area = w * h
            ar = w / h if h > 0 else 0
            # Filtrar por relación de aspecto y área
            if not (0.5 < ar < 3.0 and 15000 < area < 300000):
                continue

            # Recortar y añadir con coordenada para orden
            crop = img.crop((x, y, x + w, y + h))
            segments.append((y, crop))

        # Ordenar de arriba hacia abajo
        segments.sort(key=lambda s: s[0])
        return [seg for _, seg in segments]