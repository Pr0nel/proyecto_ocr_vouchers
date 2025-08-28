# scr/validation/validator_factory.py
"""
Módulo que define la factoría para crear instancias de validadores de imágenes.
"""
import torch
import logging
from scr.validation.ivalidator import IValidator
from scr.validation.voucher_validation import ClipValidator

class ValidatorFactory:
    """
    Factory encargada de crear instancias de componentes que implementan `IValidator`.

    Actualmente, solo soporta la creación de `ClipValidator`, pero está diseñada
    para ser extendida y poder crear otros tipos de validadores en el futuro
    basándose en la configuración proporcionada.
    """
    @staticmethod
    def create_validator(validation_config: dict, device: torch.device) -> IValidator:
        """
        Crea y devuelve una instancia de un validador de imágenes.

        La implementación específica del validador se determina (o podría determinarse en el futuro)
        a partir de `validation_config`. Actualmente, siempre crea `ClipValidator`.

        Args:
            validation_config (dict): Un diccionario con la configuración para el validador.
                Debe contener claves como 'checkpoint', 'labels', 'confidence_threshold'
                para `ClipValidator`.
            device (torch.device): El dispositivo (CPU o CUDA) donde se cargará el modelo
                                   del validador.

        Returns:
            IValidator: Una instancia de una clase que implementa la interfaz `IValidator`.

        Raises:
            ValueError: Si se especifica un tipo de validador no soportado en `validation_config`
                        (funcionalidad futura, no implementada actualmente).
            KeyError: Si faltan claves esenciales en `validation_config` para el tipo de
                      validador seleccionado.
            # Otras excepciones pueden surgir de la creación de la instancia del validador.
        """
        logger = logging.getLogger(ValidatorFactory.__name__)
        
        logger.debug(
            f"Creando ClipValidator con configuración: "
            f"checkpoint='{validation_config.get('checkpoint')}', "
            f"labels='{validation_config.get('labels')}', "
            f"confidence_threshold='{validation_config.get('confidence_threshold')}', "
            f"device='{device}'"
        )
        
        instance = ClipValidator(
            checkpoint=validation_config['checkpoint'],
            labels=validation_config['labels'],
            confidence_threshold=validation_config['confidence_threshold'],
            device=device
        )
        logger.info(f"Instancia de {type(instance).__name__} creada exitosamente.")
        return instance
