#scr/validation/voucher_validation.py
"""
Implementación de un validador de imágenes (`IValidator`) que utiliza un modelo
CLIP (Contrastive Language-Image Pre-Training) para determinar si una imagen
corresponde a un voucher.
"""
import torch
from typing import List
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import logging
from scr.validation.ivalidator import IValidator

class ClipValidator(IValidator):
    """
    Validador de vouchers basado en un modelo CLIP.

    Esta clase carga un modelo CLIP pre-entrenado y su procesador asociado para
    calcular la similitud entre una imagen dada y una serie de etiquetas de texto
    (ej. "voucher", "no voucher") y así clasificar la imagen.
    """
    def __init__(self, checkpoint: str, device: torch.device, labels: List[str], confidence_threshold: float = 0.8):
        """
        Inicializa el validador CLIP.

        Args:
            checkpoint (str): Nombre o ruta al checkpoint del modelo CLIP 
                              (ej. "openai/clip-vit-base-patch32").
            device (torch.device): Dispositivo (CPU o CUDA) donde se cargará el modelo.
            labels (List[str]): Lista de etiquetas de texto que se usarán para la clasificación
                                (ej. ["voucher", "no voucher"]).
            confidence_threshold (float, optional): Umbral de confianza para considerar una imagen 
                                                  como voucher. Defaults to 0.8.
        """
        self.device = device
        self.model  = CLIPModel.from_pretrained(checkpoint).to(device)
        self.processor  = CLIPProcessor.from_pretrained(checkpoint)
        self.labels = labels
        self.threshold = confidence_threshold
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"ClipValidator inicializado con checkpoint: {checkpoint}")
        self.logger.debug(f"Etiquetas para ClipValidator: {self.labels}")
        self.logger.debug(f"Umbral de confianza para ClipValidator: {self.threshold}")
        self.logger.debug(f"Dispositivo para ClipValidator: {self.device}")

    @staticmethod
    def _evaluar_probabilidades(
        probs: List[float], 
        labels: List[str], 
        etiqueta_objetivo: str, 
        umbral: float
    ) -> bool:
        """
        Evalúa si la probabilidad de la etiqueta objetivo en la lista de probabilidades
        cumple con el umbral especificado.

        Si `etiqueta_objetivo` no se encuentra en `labels`, se utiliza la probabilidad máxima
        en `probs` como fallback (este comportamiento es heredado de la lógica original).

        Args:
            probs (List[float]): Lista de probabilidades calculadas por el modelo,
                                 correspondientes al orden de `labels`.
            labels (List[str]): Lista de etiquetas de texto usadas para generar las `probs`.
            etiqueta_objetivo (str): La etiqueta específica cuya probabilidad nos interesa.
            umbral (float): El umbral de confianza.

        Returns:
            bool: True si la probabilidad de `etiqueta_objetivo` (o el máximo si hay fallback)
                  es mayor o igual al `umbral`, False en caso contrario. 
                  Devuelve False si `probs` está vacía.
        """
        final_prob = 0.0
        if not probs: # Manejar el caso de lista de probabilidades vacía si es posible
             return False

        try:
            # Intenta encontrar el índice de la etiqueta objetivo específica.
            idx = labels.index(etiqueta_objetivo)
            final_prob = probs[idx]
        except ValueError:
            # Si la etiqueta objetivo no está en la lista, tomar la probabilidad máxima.
            final_prob = max(probs) if probs else 0.0 
            # (Añadido 'if probs else 0.0' para evitar error en max() con lista vacía)
        
        return final_prob >= umbral

    def is_voucher(self, image: Image.Image) -> bool:
        """
        Determina si la imagen proporcionada es un voucher utilizando el modelo CLIP.

        Procesa la imagen y las etiquetas configuradas, obtiene las probabilidades
        del modelo CLIP y utiliza el método `_evaluar_probabilidades` para tomar
        la decisión final basada en el umbral de confianza.

        Args:
            image (Image.Image): La imagen (en formato PIL) a validar.

        Returns:
            bool: `True` si la imagen es clasificada como voucher, `False` en caso contrario.
        
        Raises:
            # Podría propagar excepciones de Hugging Face Transformers si el modelo
            # o el procesador tienen problemas con la imagen de entrada.
            Exception: Si ocurre un error durante el procesamiento con el modelo CLIP.
                       (Esta es una generalización, errores específicos de Transformers
                       o PyTorch podrían ocurrir).
        """
        try:
            inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            
            image_probs: List[float] = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()
            self.logger.debug(f"Probabilidades obtenidas: {image_probs}")
            
            if 'voucher' not in self.labels:
                self.logger.warning(f"La etiqueta 'voucher' no se encuentra en self.labels ({self.labels}). ClipValidator usará max(probs) como alternativa.")

            resultado_validacion = ClipValidator._evaluar_probabilidades(
                probs=image_probs,
                labels=self.labels,
                etiqueta_objetivo='voucher', 
                umbral=self.threshold
            )
            self.logger.info(f"Resultado de validación: {'ES voucher' if resultado_validacion else 'NO ES voucher'}")
            return resultado_validacion
        except Exception as e:
            self.logger.error(f"Error durante la validación de la imagen con CLIP: {e}")
            self.logger.exception("Detalles del error de validación CLIP:")
            # Dependiendo de la política de errores, podrías re-lanzar la excepción o devolver False por defecto en caso de error.
            # Por ahora, se re-lanza para no ocultar el problema.
            raise
