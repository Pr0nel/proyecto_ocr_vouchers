#scr/ocr/ocr_extractor_factory.py
"""
Módulo que define la factory para crear instancias de extractores de OCR.
"""
import logging
from scr.ocr.iocrextractor import IOCRExtractor
from scr.ocr.voucher_ocr import OCRExtractor

class OCRExtractorFactory:
    """
    Factory encargada de crear instancias de componentes que implementan `IOCRExtractor`.

    Actualmente, crea instancias de la clase `OCRExtractor`, la cual internamente
    gestiona diferentes motores de OCR (Tesseract, Donut, Textract) basados en la
    configuración proporcionada.
    """
    @staticmethod
    def create_ocr_extractor(ocr_config: dict) -> IOCRExtractor:
        """
        Crea y devuelve una instancia de un extractor de OCR.

        La configuración `ocr_config` debe contener la clave 'method' para determinar
        la estrategia de OCR que `OCRExtractor` utilizará.

        Args:
            ocr_config (dict): Un diccionario con la configuración para el extractor de OCR.
                               Debe contener la clave 'method' (ej. "tesseract", "donut", "textract")
                               y, opcionalmente, otras configuraciones específicas del método
                               (ej. 'donut_model' si `OCRExtractor` no lo carga de settings.yml).

        Returns:
            IOCRExtractor: Una instancia de `OCRExtractor` configurada con el método especificado.

        Raises:
            KeyError: Si falta la clave 'method' en `ocr_config`.
            # Otras excepciones pueden surgir de la creación de `OCRExtractor` si la configuración
            # del método o modelo específico es incorrecta (ej. modelo Donut no encontrado).
        """
        logger = logging.getLogger(OCRExtractorFactory.__name__)
        logger.debug(
            f"Creando OCRExtractor con configuración: method='{ocr_config.get('method')}'"
        )
        
        instance = OCRExtractor(
            method=ocr_config['method']
            # donut_model es manejado internamente por OCRExtractor basado en settings.yml
            # si no se pasa explícitamente y method es 'donut'.
        )
        logger.info(f"Instancia de OCRExtractor creada exitosamente con método: {ocr_config.get('method')}.")
        return instance
