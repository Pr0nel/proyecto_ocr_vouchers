#main.py
"""
Punto de entrada principal para la aplicación de procesamiento de vouchers.

Este script carga la configuración, inicializa los componentes necesarios del pipeline
(segmentador, validador, extractor de OCR) a través de sus respectivas factory,
inyecta estas dependencias en el `VoucherPipeline`, y luego ejecuta el pipeline.
Incluye configuración de logging y manejo de errores a alto nivel.
"""
import torch
from scr.pipeline.voucher_pipeline import VoucherPipeline
from scr.utils.config_loader import load_config
from scr.segmentation.segmenter_factory import SegmenterFactory
from scr.validation.validator_factory import ValidatorFactory
from scr.ocr.ocr_extractor_factory import OCRExtractorFactory
from scr.utils.logger import setup_logger

if __name__ == '__main__':
    # Cargar la configuración global de la aplicación.
    config = load_config('config/settings.yml') 
    
    # Configurar el logger principal para la aplicación.
    log_config = config.get('logging', {}) # Obtener config de logging, o dict vacío si no existe
    log_file = log_config.get('log_file', 'outputs/logs/default_main.log') # Valor por defecto
    log_level = log_config.get('log_level', 'INFO') # Valor por defecto
    logger = setup_logger('main_app', ruta_log_file=log_file, level=log_level)
    
    logger.info("=================================================")
    logger.info("Iniciando la aplicación de procesamiento de vouchers")
    logger.info("=================================================")
        
    try:
        # ----- INICIO DE LA LÓGICA PRINCIPAL QUE VA DENTRO DEL TRY -----
        
        # Determinar el dispositivo (CPU o CUDA si está disponible).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Dispositivo seleccionado: {device}")
        
        # Loguear información de configuración relevante.
        logger.info(f"Configuración de Segmentación - Modelo: {config.get('segmentation', {}).get('model_name')}")
        logger.info(f"Configuración de Validación - Checkpoint: {config.get('validation', {}).get('checkpoint')}")
        logger.info(f"Configuración de OCR - Método: {config.get('ocr', {}).get('method')}")
            
        # Preparar diccionario de directorios base.
        base_dirs = {
            'vouchers_a_segmentar': config['paths']['vouchers_a_segmentar'],
            'single_voucher': config['paths']['single_voucher'],
            'no_voucher': config['paths']['output_no_voucher_dir'],
            'validated_voucher': config['paths']['validated_voucher_dir'],
            'outputs': config['paths']['outputs_json_dir'],
        } 
        logger.debug(f"Directorios base configurados: {base_dirs}")

        # Crear instancias de los componentes usando las factory.
        logger.info("Inicializando componentes del pipeline...")
        segmenter = SegmenterFactory.create_segmenter(config['segmentation'], device)
        logger.debug(f"Instancia de Segmenter creada: {type(segmenter).__name__}")
            
        validator = ValidatorFactory.create_validator(config['validation'], device)
        logger.debug(f"Instancia de Validator creada: {type(validator).__name__}")

        ocr_extractor = OCRExtractorFactory.create_ocr_extractor(config['ocr'])
        logger.debug(f"Instancia de OCRExtractor creada: {type(ocr_extractor).__name__}")
        logger.info("Componentes del pipeline inicializados correctamente.")

        # Crear e iniciar el pipeline con las dependencias inyectadas.
        pipeline = VoucherPipeline(
            segmenter=segmenter,
            validator=validator,
            ocr_extractor=ocr_extractor,
            base_dirs=base_dirs
        )
            
        logger.info("Iniciando VoucherPipeline.run()...")
        pipeline.run()
        logger.info("VoucherPipeline.run() ha finalizado.")
        # ----- FIN DE LA LÓGICA PRINCIPAL -----
            
    except FileNotFoundError as e:
        logger.error(f"Error de archivo no encontrado: {e}. Verifica las rutas en config/settings.yml y la existencia de archivos/modelos.")
    except KeyError as e:
        logger.error(f"Error de clave no encontrada: {e}. Verifica que todas las claves necesarias estén en config/settings.yml.")
    except Exception as e:
        logger.exception(f"Ha ocurrido un error inesperado en la ejecución principal: {e}") # logger.exception incluye traceback
        
    logger.info("================================================")
    logger.info("Aplicación de procesamiento de vouchers finalizada.")
    logger.info("================================================")
