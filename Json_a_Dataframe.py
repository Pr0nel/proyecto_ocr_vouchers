import pandas as pd
import json

# Cargar JSON
path = "prueba_ocr_textract/image_05_voucher_0.json"
with open(path, encoding='utf-8') as f:
    datos = json.load(f)

# Convertir a DataFrame (asumiendo que es una lista de diccionarios)
df = pd.json_normalize(datos)  # O pd.DataFrame(datos), dependiendo de la estructura

# Mostrar la tabla
print(df)

# Guardar en Excel
df.to_excel('salida.xlsx', index=False)
