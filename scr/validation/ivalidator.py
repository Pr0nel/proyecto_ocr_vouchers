# scr/validation/ivalidator.py
"""
Define la interfaz abstracta para los componentes de validación de imágenes.
"""
from abc import ABC, abstractmethod
from PIL import Image

class IValidator(ABC):
    """
    Interfaz abstracta para un validador de imágenes.

    Los validadores concretos deben implementar el método `is_voucher` para
    determinar si una imagen dada cumple con ciertos criterios de validación
    (en este contexto, si es o no un voucher).
    """
    @abstractmethod
    def is_voucher(self, image: Image.Image) -> bool: # Usar Image.Image si PIL se importa solo como Image
        """
        Valida una imagen para determinar si es un voucher.

        Args:
            image (Image.Image): El objeto PIL.Image a validar.

        Returns:
            bool: `True` si la imagen se considera un voucher válido, `False` en caso contrario.
        
        Raises:
            NotImplementedError: Si el método no es implementado por la subclase.
            # Considerar también excepciones específicas si el procesamiento de la imagen
            # o la validación fallan de manera irrecuperable.
        """
        pass
