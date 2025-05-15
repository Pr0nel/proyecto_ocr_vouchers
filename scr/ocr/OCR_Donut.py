import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Configuración: asigna aquí la ruta de tu imagen
image_path = "/home/lenovo_ubuntu/home/PROYECTOS/test_OCR/segmentos_vouchers/vit_b_voucher_0.png"  # ← Reemplaza con la ruta de tu imagen

# Selecciona el dispositivo: GPU si está disponible, de lo contrario CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga el procesador y el modelo preentrenado afinado con CORD v2
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model.to(device)

# Carga y convierte la imagen a RGB
image = Image.open(image_path).convert("RGB")

# Preprocesa la imagen para obtener los valores de píxeles
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# Define el prompt de la tarea
task_prompt = "<s_cord-v2>"

# Tokeniza el prompt para obtener los IDs de entrada del decodificador
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

# Realiza la inferencia sin calcular gradientes
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

# Decodifica la salida generada para obtener el texto
sequence = processor.batch_decode(outputs.sequences)[0]

# Limpia el texto eliminando tokens especiales y espacios innecesarios
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").strip()

# Convierte la secuencia en un diccionario JSON estructurado
parsed_output = processor.token2json(sequence)

# Guardar la salida en un archivo JSON
output_path = "resultado.json"  # Puedes cambiar el nombre y la ubicación del archivo
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(parsed_output, f, ensure_ascii=False, indent=4)

print(f"El resultado se ha guardado en {output_path}")