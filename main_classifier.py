# main_classifier.py

import torch
from transformers import CLIPProcessor, CLIPModel
from models.clip_classifier import CLIPVoucherClassifier    # Importa la clase clasificador
from scr.utils.config_loader import load_config                 # Carga función auxiliar para leer settings.yaml
from scr.utils.logger import setup_logger

logger  = setup_logger("main_classifier",None)

# Carga la configuración desde el archivo YAML
config      = load_config("config/settings.yml")
#logger      = setup_logger("main_classifier",log_file=config["logging"]["ruta_log_file"]) # Crear archivo logs en ruta de config
model_name  = config["validation"]["checkpoint"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga única del modelo + procesador
model       = CLIPModel.from_pretrained(model_name)  # ¡aún en CPU!
processor   = CLIPProcessor.from_pretrained(model_name, use_fast=False) # use_fast=False para python

# Instanciar clasificador
classifier      = CLIPVoucherClassifier(
    model       = model,
    processor   = processor,
    labels      = config["validation"]["labels"],
    device      = device
)

"""
# Imagen a clasificar
image_path  = config["paths"]["input_dir"] #image_path = "data/input/ejemplo.jpg"

# Ejecutar predicción
prediction, scores  = classifier.predict(image_path)

# Log resultados
logger.info(f"Imagen: {image_path}")
logger.info(f"Predicción: {prediction}")
logger.info(f"Scores: {scores}")
"""

# Procesa carpeta: mueve las no-voucher a la ruta configurada
input_folder_dir        = config["paths"]["input_folder_dir"] # carpeta con imágenes a validar
output_validated_voucher_dir   = config["paths"]["output_validated_voucher_dir"]
output_no_voucher_dir   = config["paths"]["output_no_voucher_dir"]

classifier.process_folder(input_folder_dir, output_validated_voucher_dir, output_no_voucher_dir)
