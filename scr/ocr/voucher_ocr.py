#scr/ocr/voucher_ocr.py
"""
Implementación de un extractor de texto OCR (`IOCRExtractor`) que puede utilizar
diferentes motores de OCR como estrategias (Tesseract, Donut, AWS Textract).

La selección del motor de OCR se realiza durante la instanciación de la clase.
También incluye utilidades de preprocesamiento de imágenes para OCR.
"""
import io
import cv2
import boto3
import pytesseract
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import logging
from scr.utils.config_loader import load_config
from scr.ocr.iocrextractor import IOCRExtractor

class OCRExtractor(IOCRExtractor):
    """
    Extractor de texto OCR que implementa `IOCRExtractor` y gestiona múltiples
    estrategias/motores de OCR.

    Al inicializarse, se configura para usar un método de OCR específico
    (Tesseract, Donut, o AWS Textract). Carga los modelos o clientes necesarios
    para el método seleccionado.
    """
    def __init__(self, method: str = 'tesseract', donut_model: str | None = None): # Python 3.10+ type hint
        """
        Inicializa el OCRExtractor con el método de OCR especificado.

        Args:
            method (str, optional): El método de OCR a utilizar.
                Opciones válidas: "tesseract", "donut", "textract".
                Defaults to 'tesseract'.
            donut_model (str | None, optional): Nombre o ruta del modelo Donut a utilizar si `method` es "donut".
                Si es `None` y `method` es "donut", se intentará cargar desde la configuración
                en `config/settings.yaml` bajo `ocr.donut_model`.
                Defaults to None.
        """
        self.method = method
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"OCRExtractor inicializado con método: {self.method}")
        
        self.loaded_donut_model_name: str | None = None # Para almacenar el nombre del modelo Donut si se carga

        if method == 'donut':
            cfg = load_config('config/settings.yaml')
            self.loaded_donut_model_name = donut_model or cfg['ocr']['donut_model'] 
            self.processor = DonutProcessor.from_pretrained(self.loaded_donut_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.loaded_donut_model_name)
            self.logger.info(f"Usando modelo Donut: {self.loaded_donut_model_name}")
        elif method == 'textract':
            self.textract = boto3.client('textract')
            self.logger.info("Cliente AWS Textract inicializado.")
        elif self.method == 'tesseract':
            self.logger.info("Usando Tesseract OCR.")

    @staticmethod
    def preprocesar_imagen(image: Image.Image) -> Image.Image:
        """
        Aplica una serie de pasos de preprocesamiento a una imagen para mejorar
        potencialmente los resultados del OCR, especialmente para Tesseract.

        Los pasos incluyen: conversión a RGB, luego a escala de grises, binarización
        con el método de Otsu, y un filtro de desenfoque mediano.

        Args:
            image (Image.Image): La imagen (en formato PIL) a preprocesar.

        Returns:
            Image.Image: La imagen preprocesada (en formato PIL y binarizada).
        """
        arr = np.array(image.convert('RGB'))
        gris = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria = cv2.medianBlur(binaria, 3)
        return Image.fromarray(binaria)

    def extract(self, image: Image.Image) -> str:
        """
        Extrae texto de la imagen utilizando el método de OCR configurado.

        Delega la extracción a la implementación específica (Tesseract, Donut, Textract)
        basada en el `method` establecido durante la inicialización.

        Args:
            image (Image.Image): La imagen (en formato PIL) de la cual extraer texto.

        Returns:
            str: El texto extraído. Puede ser una cadena vacía si no se detecta texto.

        Raises:
            ValueError: Si el `method` de OCR configurado no es soportado.
            # Otras excepciones pueden ser propagadas por los motores de OCR subyacentes
            # (ej. errores de Tesseract, problemas de red con Textract, errores de modelo Donut).
        """
        
        if self.method == 'tesseract':
            self.logger.debug("Procesando con Tesseract...")
            img_proc = OCRExtractor.preprocesar_imagen(image)
            texto_extraido = pytesseract.image_to_string(img_proc, config = '--oem 3 --psm 6')
            self.logger.info(f"Tesseract OCR completado. Texto extraído (primeros 50 caracteres): '{texto_extraido[:50]}...'")
            return texto_extraido
        
        elif self.method == 'donut':
            self.logger.debug("Procesando con Donut...")
            prompt  = "<s_cord-v2>" #tablas de voucher
            decoder_input_ids = self.processor.tokenizer(
                                    prompt,
                                    add_special_tokens=False,
                                    return_tensors="pt"
                                ).input_ids.to(self.model.device)
            inputs = self.processor(image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.model.device)
            
            # Usar max_length del modelo si está disponible, de lo contrario un valor por defecto razonable
            # El uso de decoder_start_token_id como max_length es inusual y podría ser un error.
            # Un valor más común sería self.model.config.max_length o un número grande.
            # Por ahora, se mantiene la lógica del diff anterior, pero es un punto a revisar.
            max_len = getattr(self.model.config, 'max_length', 
                              getattr(self.model.config, 'decoder_start_token_id', 512)) # Fallback a 512 si no se encuentra

            outputs = self.model.generate(
                        pixel_values=pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=max_len, 
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                        )
            sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            self.logger.info(f"Donut OCR completado. Texto extraído (primeros 50 caracteres): '{sequence[:50]}...'")
            return sequence
        
        elif self.method == 'textract':
            self.logger.debug("Procesando con AWS Textract...")
            buf = io.BytesIO()
            image.convert('RGB').save(buf, format='JPEG')
            buf.seek(0)
            response = self.textract.detect_document_text(
                Document={'Bytes': buf.read()}
            )
            lines = [item['Text'] for item in response.get('Blocks', []) if item['BlockType'] == 'LINE']
            texto_final = ' '.join(lines)
            self.logger.info(f"AWS Textract OCR completado. Texto extraído (primeros 50 caracteres): '{texto_final[:50]}...'")
            return texto_final
        else:
            self.logger.error(f"Método OCR no soportado intentado: {self.method}")
            raise ValueError(f"Método OCR no soportado: {self.method}")
