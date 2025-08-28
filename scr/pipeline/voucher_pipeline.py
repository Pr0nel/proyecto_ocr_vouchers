#scr/pipeline/voucher_pipeline.py
"""
Módulo que define el pipeline principal para el procesamiento de vouchers.

La clase `VoucherPipeline` orquesta las etapas de segmentación, validación
y extracción de texto (OCR) de los vouchers.
"""
import json
import torch
import shutil
from PIL import Image
from pathlib import Path
import logging
from scr.segmentation.isegmenter import ISegmenter
from scr.validation.ivalidator import IValidator
from scr.ocr.iocrextractor import IOCRExtractor
#from scr.utils.text_processing import remover_espacios_extra, convertir_a_minusculas

class VoucherPipeline:
    """
    Orquesta el pipeline completo para el procesamiento de imágenes de vouchers.

    Recibe instancias de los componentes necesarios (segmentador, validador, extractor de OCR)
    a través de inyección de dependencias en su constructor. El método `run` ejecuta
    la secuencia de procesamiento.
    """
    def __init__(self,
                 segmenter: ISegmenter,
                 validator: IValidator,
                 ocr_extractor: IOCRExtractor,
                 base_dirs: dict):
        """
        Inicializa el VoucherPipeline con sus dependencias y configuración de directorios.

        Args:
            segmenter (ISegmenter): Instancia de un segmentador de imágenes.
            validator (IValidator): Instancia de un validador de imágenes.
            ocr_extractor (IOCRExtractor): Instancia de un extractor de OCR.
            base_dirs (dict): Diccionario con las rutas base para los directorios
                              de entrada y salida (ej. 'vouchers_a_segmentar', 
                              'single_voucher', 'validated_voucher_dir', 
                              'output_no_voucher_dir', 'outputs_json_dir').
        
        Raises:
            OSError: Si hay un problema de permisos o de otro tipo al crear los directorios base.
        """
        self.segmenter = segmenter
        self.validator = validator
        self.ocr_extractor = ocr_extractor
        self.dirs = {k: Path(v) for k, v in base_dirs.items()}
        
        self.logger = logging.getLogger(self.__class__.__name__)

        # Crear directorios si no existen
        for dir_key, dir_path in self.dirs.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Directorio '{dir_key}' ({dir_path}) asegurado/creado.")
            except OSError as e:
                self.logger.error(f"No se pudo crear el directorio {dir_path} para '{dir_key}': {e}")
                raise # Relanzar si falla la creación de un directorio base.
        
        self.logger.info("VoucherPipeline inicializado con sus componentes y directorios configurados.")

    def run(self):
        """
        Ejecuta el pipeline completo de procesamiento de vouchers.

        El proceso general es:
        1. Lee imágenes del directorio 'vouchers_a_segmentar'.
        2. Para cada imagen, segmenta posibles vouchers y los guarda en 'single_voucher'.
           Maneja errores por archivo durante la segmentación.
        3. Lee cada segmento guardado de 'single_voucher'.
        4. Valida si el segmento es un voucher.
        5. Si es válido:
            a. Lo mueve a 'validated_voucher_dir'.
            b. Extrae texto usando OCR.
            c. Limpia el texto extraído.
            d. Guarda el texto crudo y limpio en un archivo JSON en 'outputs_json_dir'.
        6. Si no es válido, lo mueve a 'output_no_voucher_dir'.
           Maneja errores por archivo durante la validación y OCR.
        """
        self.logger.info("Iniciando el procesamiento del pipeline de vouchers...")
        
        # Fase de Segmentación
        self.logger.info("--- Iniciando Fase de Segmentación ---")
        for img_file in self.dirs['vouchers_a_segmentar'].iterdir():
            try:
                self.logger.info(f"Procesando archivo de imagen principal: {img_file.name}")
                segments = self.segmenter.segment(img_file)
                self.logger.info(f"Encontrados {len(segments)} segmentos en {img_file.name}.")
                for idx, seg in enumerate(segments):
                    seg_path = self.dirs['single_voucher'] / f"{img_file.stem}_voucher_{idx}.png"
                    self.logger.debug(f"Guardando segmento {idx} de {img_file.name} en: {seg_path}")
                    seg.save(seg_path)
            except Exception as e:
                self.logger.error(f"Error procesando el archivo principal {img_file.name} durante la segmentación: {e}")
                self.logger.exception("Detalles del error de segmentación:")
        self.logger.info("--- Fase de Segmentación Finalizada ---")

        # Fase de Validación y OCR
        self.logger.info("--- Iniciando Fase de Validación y OCR ---")
        for img_file in list(self.dirs['single_voucher'].iterdir()): # Convertir a lista para evitar problemas si se mueven archivos
            try:
                self.logger.info(f"Procesando segmento individual: {img_file.name}")
                img = Image.open(img_file)
                
                if self.validator.is_voucher(img):
                    self.logger.info(f"Segmento {img_file.name} VALIDADO como voucher.")
                    dest = self.dirs['validated_voucher'] / f"{img_file.stem}_v{img_file.suffix}"
                    self.logger.info(f"Moviendo {img_file.name} a {dest}")
                    shutil.move(str(img_file), dest)
                    
                    self.logger.info(f"Extrayendo OCR de {dest.name}...")
                    raw_text = self.ocr_extractor.extract(Image.open(dest)) 
                    self.logger.debug(f"Texto crudo extraído de {dest.name}: '{raw_text[:100]}...'")
                    
                    # Aplicar funciones de limpieza
                    #text_cleaned_spaces = remover_espacios_extra(raw_text)
                    #cleaned_text = convertir_a_minusculas(text_cleaned_spaces)
                    # (Si se añaden más funciones de limpieza, se aplicarían secuencialmente aquí)
                    #self.logger.debug(f"Texto limpio para {dest.name}: '{cleaned_text[:100]}...'")
                    
                    json_path = self.dirs['outputs'] / f"{dest.stem}.json"
                    output_data = {
                        'raw_text': raw_text,
                        #'cleaned_text': cleaned_text 
                        # Podrías añadir más campos si fuera necesario, como 'timestamp', etc.
                    }
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=4)
                    self.logger.info(f"Resultado OCR para {dest.name} guardado en {json_path}")
                else:
                    self.logger.info(f"Segmento {img_file.name} NO VALIDADO como voucher.")
                    dest = self.dirs['no_voucher'] / img_file.name
                    self.logger.info(f"Moviendo {img_file.name} a {dest}")
                    shutil.move(str(img_file), dest)
            except Exception as e:
                self.logger.error(f"Error procesando el segmento {img_file.name} durante la validación/OCR: {e}")
                self.logger.exception("Detalles del error de validación/OCR:")
        self.logger.info("--- Fase de Validación y OCR Finalizada ---")
        
        self.logger.info("Procesamiento del pipeline de vouchers finalizado.")
