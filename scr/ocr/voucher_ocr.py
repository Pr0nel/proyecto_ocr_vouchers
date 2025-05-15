#scr/ocr/voucher_ocr.py
import io
import cv2
import boto3
import pytesseract
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from scr.utils.config_loader import load_config

class OCRExtractor:
    def __init__(self, method: str = 'tesseract', donut_model: str = None):
        self.method = method
        # Carga configuración para modelo Donut si aplica
        if method == 'donut':
            cfg             = load_config('config/settings.yaml')
            model_name      = donut_model or cfg['ocr']['donut_model']
            self.processor  = DonutProcessor.from_pretrained(model_name)
            self.model      = VisionEncoderDecoderModel.from_pretrained(model_name)
        elif method == 'textract':
            # Cliente de Textract
            self.textract   = boto3.client('textract')

    def preprocesar_imagen(self, image: Image.Image):
        arr         = np.array(image.convert('RGB'))
        gris        = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, binaria  = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria     = cv2.medianBlur(binaria, 3)
        return Image.fromarray(binaria)

    def extract(self, image: Image.Image):
        if self.method == 'tesseract':
            img_proc = self.preprocesar_imagen(image)
            return pytesseract.image_to_string(img_proc, config = '--oem 3 --psm 6')
        
        elif self.method == 'donut':
            prompt  = "<s_cord-v2>"
            decoder_input_ids   = self.processor.tokenizer(
                                    prompt,
                                    add_special_tokens = False,
                                    return_tensors = "pt"
                                    ).input_ids.to(self.model.device)
            inputs  = self.processor(image, return_tensors = "pt")
            pixel_values        = inputs.pixel_values.to(self.model.device)
            outputs = self.model.generate(
                        pixel_values= pixel_values,
                        decoder_input_ids   = decoder_input_ids,
                        max_length  = 768,
                        early_stopping      = True,
                        pad_token_id= self.processor.tokenizer.pad_token_id,
                        eos_token_id= self.processor.tokenizer.eos_token_id
                        )
            return self.processor.batch_decode(outputs, skip_special_tokens = True)[0]
        
        elif self.method == 'textract':
            # Convierte la imagen PIL a bytes
            buf     = io.BytesIO()
            image.convert('RGB').save(buf, format = 'JPEG')
            buf.seek(0)
            response= self.textract.detect_document_text(
                Document= {'Bytes': buf.read()}
            )
            # Concatenar texto extraído
            lines   = [item['Text'] for item in response.get('Blocks', []) if item['BlockType']=='LINE']
            return ' '.join(lines)
        else:
            raise ValueError(f"Método OCR no soportado: {self.method}")
