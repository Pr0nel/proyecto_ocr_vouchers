import os
import json
import torch
import logging
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Configuración básica del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ocr_proceso.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def cargar_modelo(modelo_nombre: str, dispositivo: str):
    """
    Carga el modelo y el procesador preentrenados.

    Args:
        modelo_nombre (str): Nombre del modelo preentrenado.
        dispositivo (str): Dispositivo a utilizar ('cuda' o 'cpu').

    Returns:
        tuple: Procesador y modelo cargados.
    """
    try:
        processor = DonutProcessor.from_pretrained(modelo_nombre)
        model = VisionEncoderDecoderModel.from_pretrained(modelo_nombre)
        model.to(dispositivo)
        logging.info(f"Modelo '{modelo_nombre}' cargado en {dispositivo}.")
        return processor, model
    except Exception as e:
        logging.exception(f"Error al cargar el modelo '{modelo_nombre}': {e}")
        raise

def procesar_imagen(ruta_imagen: str):
    """
    Abre y convierte una imagen a formato RGB.

    Args:
        ruta_imagen (str): Ruta de la imagen a procesar.

    Returns:
        Image: Imagen en formato RGB.
    """
    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
        logging.info(f"Imagen '{ruta_imagen}' cargada y convertida a RGB.")
        return imagen
    except Exception as e:
        logging.exception(f"Error al procesar la imagen '{ruta_imagen}': {e}")
        raise

def generar_salida(processor, model, imagen, dispositivo: str, prompt: str):
    """
    Genera la salida del modelo para una imagen dada.

    Args:
        processor: Procesador del modelo.
        model: Modelo preentrenado.
        imagen: Imagen a procesar.
        dispositivo (str): Dispositivo a utilizar ('cuda' o 'cpu').
        prompt (str): Prompt de entrada para el modelo.

    Returns:
        str: Secuencia generada por el modelo.
    """
    try:
        pixel_values = processor(imagen, return_tensors="pt").pixel_values.to(dispositivo)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dispositivo)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        secuencia = processor.batch_decode(outputs.sequences)[0]
        secuencia = secuencia.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").strip()
        logging.info("Secuencia generada exitosamente.")
        return secuencia
    except Exception as e:
        logging.exception(f"Error al generar la salida del modelo: {e}")
        raise

def guardar_json(salida_json, ruta_salida: str):
    """
    Guarda la salida en formato JSON en la ruta especificada.

    Args:
        salida_json: Datos a guardar en formato JSON.
        ruta_salida (str): Ruta del archivo de salida.
    """
    try:
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(salida_json, f, ensure_ascii=False, indent=4)
        logging.info(f"Archivo JSON guardado en '{ruta_salida}'.")
    except Exception as e:
        logging.exception(f"Error al guardar el archivo JSON '{ruta_salida}': {e}")
        raise

def main():
    carpeta_imagenes = "prueba_segmentos_vouchers"
    carpeta_salida_json = "prueba_ocr_vouchers"
    os.makedirs(carpeta_salida_json, exist_ok=True)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo_nombre = "naver-clova-ix/donut-base-finetuned-cord-v2"
    prompt = "<s_cord-v2>"

    try:
        processor, model = cargar_modelo(modelo_nombre, dispositivo)
    except Exception as e:
        logging.critical("No se pudo cargar el modelo. Terminando la ejecución.")
        return

    for nombre_archivo in os.listdir(carpeta_imagenes):
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
            try:
                imagen = procesar_imagen(ruta_imagen)
                secuencia = generar_salida(processor, model, imagen, dispositivo, prompt)
                salida_json = processor.token2json(secuencia)
                nombre_json = os.path.splitext(nombre_archivo)[0] + ".json"
                ruta_salida = os.path.join(carpeta_salida_json, nombre_json)
                guardar_json(salida_json, ruta_salida)
            except Exception as e:
                logging.error(f"Error al procesar el archivo '{nombre_archivo}': {e}")
                continue

if __name__ == "__main__":
    main()
