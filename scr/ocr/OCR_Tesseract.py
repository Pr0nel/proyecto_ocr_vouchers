import os
import json
import cv2
import pytesseract
import logging

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("json_ocr_proceso.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def preprocesar_imagen(imagen):
    """Aplica preprocesamiento a la imagen para mejorar el OCR."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

def realizar_ocr(imagen):
    """Realiza OCR en la imagen preprocesada."""
    config = '--oem 3 --psm 6'
    texto = pytesseract.image_to_string(imagen, config=config)
    return texto.strip()

def guardar_resultado(nombre_archivo, texto, carpeta_salida):
    """Guarda el texto extraído en un archivo JSON."""
    resultado = {
        "imagen": nombre_archivo,
        "texto_extraido": texto
    }
    nombre_json = os.path.splitext(nombre_archivo)[0] + ".json"
    ruta_json = os.path.join(carpeta_salida, nombre_json)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ Guardado: {nombre_json}")

def main():
    carpeta_segmentos = "segmentos_vouchers"
    carpeta_salida_json = "json_ocr_vouchers"
    os.makedirs(carpeta_salida_json, exist_ok=True)

    for nombre_imagen in sorted(os.listdir(carpeta_segmentos)):
        ruta_imagen = os.path.join(carpeta_segmentos, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            logging.warning(f"⚠️ No se pudo leer: {nombre_imagen}")
            continue

        try:
            imagen_procesada = preprocesar_imagen(imagen)
            texto = realizar_ocr(imagen_procesada)
            guardar_resultado(nombre_imagen, texto, carpeta_salida_json)
        except Exception as e:
            logging.error(f"Error al procesar {nombre_imagen}: {e}")

if __name__ == "__main__":
    main()
