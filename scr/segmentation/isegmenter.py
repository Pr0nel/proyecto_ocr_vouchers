#scr/segmentation/isegmenter.py
"""
Define la interfaz abstracta para los componentes de segmentación de imágenes.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from PIL import Image

class ISegmenter(ABC):
    """
    Interfaz abstracta para un segmentador de imágenes.

    Los segmentadores concretos deben implementar el método `segment` para
    identificar y extraer regiones de interés (segmentos) de una imagen dada.
    """

    @abstractmethod
    def segment(self, image_path: Path) -> List[Image.Image]:
        """
        Segmenta una imagen para extraer regiones de interés.

        Args:
            image_path (Path): La ruta al archivo de imagen a segmentar.

        Returns:
            List[Image.Image]: Una lista de objetos PIL.Image, donde cada imagen
                               es un segmento extraído de la imagen original.
                               La lista puede estar vacía si no se encuentran segmentos válidos.
        
        Raises:
            NotImplementedError: Si el método no es implementado por la subclase.
            # Considerar también excepciones específicas si la carga de la imagen falla
            # o si ocurre un error irrecuperable durante la segmentación.
            # Por ejemplo: FileNotFoundError, Exception
        """
        pass
