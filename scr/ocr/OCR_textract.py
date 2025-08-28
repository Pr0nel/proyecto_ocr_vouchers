import os
import json
import boto3
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ocr_textract.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Usar perfil de AWS CLI (opcional)
# os.environ["AWS_PROFILE"] = "default"  # o el nombre del perfil

# O usar variables de entorno directamente
os.environ["AWS_ACCESS_KEY_ID"] = 
os.environ["AWS_SECRET_ACCESS_KEY"] = 
# os.environ["AWS_SESSION_TOKEN"] = "tu_token_opcional"

def procesar_imagen_con_textract(ruta_imagen: str):
    """
    Procesa una imagen con Amazon Textract.

    Args:
        ruta_imagen (str): Ruta del archivo de imagen.

    Returns:
        dict: Resultado JSON del OCR.
    """
    try:
        with open(ruta_imagen, "rb") as imagen:
            cliente = boto3.client("textract", region_name="us-east-1")
            respuesta = cliente.analyze_document(
                Document={'Bytes': imagen.read()},
                FeatureTypes=["TABLES", "FORMS"]
            )
            logging.info(f"Procesada imagen: {ruta_imagen}")
            return respuesta
    except Exception as e:
        logging.exception(f"Error al procesar con Textract '{ruta_imagen}': {e}")
        raise

def guardar_json(salida_json, ruta_salida: str):
    """
    Guarda el JSON en disco.

    Args:
        salida_json: Diccionario JSON.
        ruta_salida (str): Ruta del archivo .json.
    """
    try:
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(salida_json, f, ensure_ascii=False, indent=4)
        logging.info(f"Archivo JSON guardado en '{ruta_salida}'.")
    except Exception as e:
        logging.exception(f"Error al guardar JSON: {e}")
        raise

def main():
    carpeta_imagenes = "prueba_segmentos_vouchers"
    carpeta_salida_json = "prueba_ocr_textract"
    os.makedirs(carpeta_salida_json, exist_ok=True)

    for nombre_archivo in os.listdir(carpeta_imagenes):
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
            try:
                salida_textract = procesar_imagen_con_textract(ruta_imagen)
                nombre_json = os.path.splitext(nombre_archivo)[0] + ".json"
                ruta_salida = os.path.join(carpeta_salida_json, nombre_json)
                guardar_json(salida_textract, ruta_salida)
            except Exception as e:
                logging.error(f"Error al procesar '{nombre_archivo}': {e}")
                continue

if __name__ == "__main__":
    main()
