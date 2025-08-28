#scr/segmentation/segmenter_factory.py
"""
Módulo que define la factory para crear instancias de segmentadores de imágenes.
"""
import torch
import logging
from scr.segmentation.isegmenter import ISegmenter
from scr.segmentation.voucher_segmentation import SamSegmenter

class SegmenterFactory:
    """
    Factory encargada de crear instancias de componentes que implementan `ISegmenter`.

    Actualmente, solo soporta la creación de `SamSegmenter`, pero está diseñada
    para ser extendida y poder crear otros tipos de segmentadores en el futuro
    basándose en la configuración proporcionada.
    """
    @staticmethod
    def create_segmenter(segmentation_config: dict, device: torch.device) -> ISegmenter:
        """
        Crea y devuelve una instancia de un segmentador de imágenes.

        La implementación específica del segmentador se determina (o podría determinarse en el futuro)
        a partir de `segmentation_config`. Actualmente, siempre crea `SamSegmenter`.

        Args:
            segmentation_config (dict): Un diccionario con la configuración para el segmentador.
                Debe contener claves como 'model_name' y 'checkpoint' para `SamSegmenter`.
            device (torch.device): El dispositivo (CPU o CUDA) donde se cargará el modelo
                                   del segmentador.

        Returns:
            ISegmenter: Una instancia de una clase que implementa la interfaz `ISegmenter`.

        Raises:
            ValueError: Si se especifica un tipo de segmentador no soportado en `segmentation_config`
                        (funcionalidad futura, no implementada actualmente).
            KeyError: Si faltan claves esenciales en `segmentation_config` para el tipo de
                      segmentador seleccionado.
            # Otras excepciones pueden surgir de la creación de la instancia del segmentador
            # (ej. si el checkpoint del modelo no se encuentra).
        """
        logger = logging.getLogger(SegmenterFactory.__name__)
        
        logger.debug(
            f"Creando SamSegmenter con configuración: "
            f"model_name='{segmentation_config.get('model_name')}', "
            f"checkpoint='{segmentation_config.get('checkpoint')}', "
            f"device='{device}'"
        )
        
        # Lógica actual: siempre crea SamSegmenter
        # Ejemplo de cómo podría extenderse (comentado):
        # segmenter_type = segmentation_config.get('type', 'sam')
        # if segmenter_type == 'sam':
        #     instance = SamSegmenter( # ... )
        # elif segmenter_type == 'otro_tipo':
        #     # instance = OtroSegmenter(...)
        #     pass 
        # else:
        #     logger.error(f"Tipo de segmentador no soportado: {segmenter_type}")
        #     raise ValueError(f"Tipo de segmentador no soportado: {segmenter_type}")

        instance = SamSegmenter(
            model_name=segmentation_config['model_name'],
            checkpoint=segmentation_config['checkpoint'],
            device=device
        )
        logger.info(f"Instancia de {type(instance).__name__} creada exitosamente.")
        return instance
