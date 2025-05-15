import os
import json
import cv2
import pytesseract

# === CONFIGURACIÓN ===
carpeta_segmentos = "segmentos_vouchers"
carpeta_salida_json = "json_ocr_vouchers"

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida_json, exist_ok=True)

# === PROCESAR CADA IMAGEN ===
for nombre_imagen in sorted(os.listdir(carpeta_segmentos)):
    ruta_imagen = os.path.join(carpeta_segmentos, nombre_imagen)
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print(f"⚠️ No se pudo leer: {nombre_imagen}")
        continue

    texto = pytesseract.image_to_string(imagen).strip()

    # Estructura del JSON
    resultado = {
        "imagen": nombre_imagen,
        "texto_extraido": texto
    }

    # Guardar archivo JSON individual
    nombre_json = os.path.splitext(nombre_imagen)[0] + ".json"
    ruta_json = os.path.join(carpeta_salida_json, nombre_json)

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)

    print(f"✅ Guardado: {nombre_json}")
