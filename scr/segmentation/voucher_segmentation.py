#scr/segmentation/voucher_segmentation.py
"""
Implementación de la segmentación de imágenes utilizando el modelo SAM (Segment Anything Model).

Este módulo proporciona la clase `SamSegmenter` que implementa la interfaz `ISegmenter`,
así como funciones de utilidad para el procesamiento y limpieza de máscaras de segmentación.
"""
from pathlib import Path
from typing import List
import cv2
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scr.segmentation.isegmenter import ISegmenter
import logging

def limpiar_mascara(mask: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    Aplica operaciones morfológicas con un kernel de tamaño adaptativo para limpiar el ruido de una máscara binaria.

    Args:
        mask (np.ndarray): La máscara binaria (array de NumPy) a limpiar.
        img_shape (tuple): Tupla con las dimensiones (alto, ancho) de la imagen original,
                           usada para calcular el tamaño adaptativo del kernel.

    Returns:
        np.ndarray: La máscara binaria limpiada.
    """
    kernel_size = (max(3, img_shape[0] // 50), max(3, img_shape[1] // 50))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def _procesar_mascara_individual(
    mascara_sam: dict, 
    img_pil_original: Image.Image, # Pasar la imagen PIL original completa
    min_raw_count: int = 1000, 
    ar_range: tuple[float, float] = (0.5, 3.0),
    area_range: tuple[int, int] = (15000, 300000)
) -> tuple[int, Image.Image] | None: # Python 3.10+ para ' | None'
    """
    Procesa una única máscara generada por SAM, la limpia, y aplica filtros geométricos y de área.

    Si la máscara procesada cumple con los criterios, recorta el segmento correspondiente
    de la imagen original.

    Args:
        mascara_sam (dict): Un diccionario que representa una máscara individual, 
                            tal como lo devuelve el generador de máscaras de SAM.
                            Debe contener la clave 'segmentation'.
        img_pil_original (Image.Image): La imagen original completa (en formato PIL)
                                        de la cual se extraerá el segmento.
        min_raw_count (int, optional): Número mínimo de píxeles no cero en la máscara 'raw' 
                                       para considerarla inicialmente. Defaults to 1000.
        ar_range (tuple[float, float], optional): Tupla (min_aspect_ratio, max_aspect_ratio) 
                                                  para filtrar los segmentos. Defaults to (0.5, 3.0).
        area_range (tuple[int, int], optional): Tupla (min_area, max_area) en píxeles 
                                                para filtrar los segmentos. Defaults to (15000, 300000).

    Returns:
        tuple[int, Image.Image] | None: Una tupla `(coordenada_y, imagen_recortada_PIL)`
                                        si la máscara es válida y el segmento se extrae.
                                        `None` si la máscara no cumple los criterios de filtrado.
    """
    img_h, img_w = img_pil_original.height, img_pil_original.width

    raw = (mascara_sam['segmentation'].astype(np.uint8)) * 255
    if cv2.countNonZero(raw) < min_raw_count:
        return None

    clean = limpiar_mascara(raw, (img_h, img_w))
    x, y, w, h = cv2.boundingRect(clean)

    # Evitar segmentos de ancho o alto cero que causarían error en crop o división por cero
    if w == 0 or h == 0:
        return None
            
    area = w * h
    ar = w / h

    if not (ar_range[0] < ar < ar_range[1] and area_range[0] < area < area_range[1]):
        return None

    crop = img_pil_original.crop((x, y, x + w, y + h))
    return (y, crop)

class SamSegmenter(ISegmenter):
    """
    Implementación de `ISegmenter` que utiliza el modelo SAM (Segment Anything Model)
    para realizar la segmentación automática de imágenes.

    Esta clase carga un modelo SAM específico y lo utiliza para generar máscaras
    de segmentación, las cuales son luego procesadas y filtradas para extraer
    los segmentos relevantes (presumiblemente vouchers).
    Incluye un mecanismo de caché para las máscaras generadas.
    """
    def __init__(self, model_name: str, checkpoint: str, device: torch.device):
        """
        Inicializa el segmentador SAM.

        Args:
            model_name (str): El nombre del tipo de modelo SAM a cargar (ej. "vit_b").
            checkpoint (str): La ruta al archivo de checkpoint del modelo SAM.
            device (torch.device): El dispositivo (CPU o CUDA) donde se cargará el modelo.
        """
        self.device = device
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
        self.masks_cache = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"SamSegmenter inicializado con modelo: {model_name}, checkpoint: {checkpoint}")
        self.logger.debug(f"Dispositivo para SamSegmenter: {self.device}")

    def segment(self, image_path: Path) -> List[Image.Image]:
        """
        Segmenta la imagen especificada utilizando el modelo SAM.

        Carga la imagen, genera máscaras de segmentación (utilizando caché si está disponible),
        procesa y filtra cada máscara individualmente, y devuelve una lista de los
        segmentos válidos extraídos como objetos PIL.Image.

        Args:
            image_path (Path): La ruta a la imagen a segmentar.

        Returns:
            List[Image.Image]: Una lista de imágenes PIL representando los segmentos válidos.
                               Puede estar vacía si no se encuentran segmentos o si la imagen
                               no se puede procesar.
        
        Raises:
            FileNotFoundError: Si `image_path` no existe.
            PIL.UnidentifiedImageError: Si `image_path` no es un archivo de imagen válido.
            Exception: Otras excepciones pueden surgir de la carga del modelo SAM o
                       durante la generación de máscaras si hay problemas con los datos
                       o el modelo. (Por ejemplo, problemas al leer el checkpoint de SAM,
                       o si la imagen de entrada está corrupta de una manera que PIL.Image.open
                       no detecta pero sí el procesamiento interno de SAM).
        """
        self.logger.info(f"Iniciando segmentación para imagen: {image_path.name}")
        try:
            img_pil_original = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            self.logger.error(f"Archivo de imagen no encontrado en: {image_path}")
            raise # Re-lanzar la excepción para que sea manejada más arriba si es necesario
        except Image.UnidentifiedImageError: # Corregido el nombre de la excepción
            self.logger.error(f"No se pudo identificar o abrir el archivo de imagen en: {image_path}")
            raise

        # Generar o recuperar máscaras
        if image_path in self.masks_cache:
            self.logger.debug(f"Usando máscaras de caché para: {image_path.name}")
            masks = self.masks_cache[image_path]
        else:
            self.logger.debug(f"Generando nuevas máscaras para: {image_path.name}")
            # Nota: mask_generator.generate espera un array numpy
            try:
                masks = self.mask_generator.generate(np.array(img_pil_original))
                self.logger.debug(f"Generadas {len(masks)} máscaras SAM (antes de filtrar) para {image_path.name}")
                self.masks_cache[image_path] = masks
            except Exception as e: # Captura de errores durante la generación de máscaras
                self.logger.error(f"Error durante la generación de máscaras SAM para {image_path.name}: {e}")
                self.logger.exception("Detalles del error de generación de máscaras SAM:")
                return [] # Devolver lista vacía si la generación de máscaras falla

        # Usar la nueva función para procesar y filtrar las máscaras
        # Los parámetros de filtrado (min_raw_count, ar_range, area_range)
        # podrían ser parte de la configuración de SamSegmenter en el futuro
        # y pasados aquí desde self.alguna_configuracion si fuera necesario.
        # Por ahora, usamos los valores por defecto de la firma de _procesar_mascara_individual.
        
        segmentos_con_y = [
            resultado for m in masks
            if (resultado := _procesar_mascara_individual(m, img_pil_original)) is not None
        ]

        # Ordenar de arriba hacia abajo por la coordenada y
        segmentos_con_y.sort(key=lambda s: s[0])
        
        self.logger.info(f"Segmentación finalizada para {image_path.name}. Encontrados {len(segmentos_con_y)} segmentos válidos.")
        # Devolver solo las imágenes recortadas
        return [seg_img for _, seg_img in segmentos_con_y]
