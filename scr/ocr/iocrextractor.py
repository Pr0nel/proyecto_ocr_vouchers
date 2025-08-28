#scr/ocr/iocrextractor.py
"""
Define la interfaz abstracta para los componentes de extracción de texto (OCR).
"""
from abc import ABC, abstractmethod
from PIL import Image

class IOCRExtractor(ABC):
    """
    Interfaz abstracta para un extractor de texto mediante OCR (Reconocimiento Óptico de Caracteres).

    Los extractores de OCR concretos deben implementar el método `extract` para
    procesar una imagen y devolver el texto contenido en ella.
    """

    @abstractmethod
    def extract(self, image: Image.Image) -> str: # Usar Image.Image si PIL se importa solo como Image
        """
        Extrae texto de una imagen utilizando una técnica de OCR.

        Args:
            image (Image.Image): El objeto PIL.Image del cual se extraerá el texto.

        Returns:
            str: Una cadena de texto con el contenido extraído de la imagen.
                 Puede ser una cadena vacía si no se detecta texto o si la imagen
                 no se puede procesar.
        
        Raises:
            NotImplementedError: Si el método no es implementado por la subclase.
            # Considerar también excepciones específicas si el motor de OCR
            # falla o si la imagen de entrada es inválida.
        """
        pass
