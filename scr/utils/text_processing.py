# scr/utils/text_processing.py
"""
Módulo con funciones de utilidad para el procesamiento y limpieza de cadenas de texto.

Estas funciones están diseñadas para ser puras en la medida de lo posible,
tomando una cadena de entrada y devolviendo una nueva cadena con las transformaciones aplicadas.
"""

# import re # Descomentar si se añade la función de ejemplo

def remover_espacios_extra(texto: str) -> str:
    """
    Elimina espacios múltiples, tabulaciones, saltos de línea y espacios al inicio/final de una cadena.

    Reemplaza secuencias de espacios en blanco con un solo espacio simple.

    Args:
        texto (str): La cadena de texto a procesar.

    Returns:
        str: La cadena de texto con los espacios extra eliminados. 
             Devuelve una cadena vacía si la entrada es None.
    """
    if texto is None:
        return ""
    return " ".join(texto.split())

def convertir_a_minusculas(texto: str) -> str:
    """
    Convierte todos los caracteres alfabéticos de una cadena a minúsculas.

    Args:
        texto (str): La cadena de texto a convertir.

    Returns:
        str: La cadena de texto en minúsculas.
             Devuelve una cadena vacía si la entrada es None.
    """
    if texto is None:
        return ""
    return texto.lower()

# Ejemplo de función adicional que podría añadirse:
# def remover_puntuacion(texto: str, conservar_espacios: bool = False) -> str:
#     """
#     Elimina los signos de puntuación comunes de una cadena.
#
#     Args:
#         texto (str): La cadena de texto a procesar.
#         conservar_espacios (bool): Si es True, los espacios no se eliminan junto con la puntuación.
#                                   Si es False, la puntuación se elimina y los espacios pueden consolidarse.
#
#     Returns:
#         str: La cadena sin puntuación.
#              Devuelve una cadena vacía si la entrada es None.
#     """
#     if texto is None:
#         return ""
#     # Definir qué se considera puntuación
#     # import string
#     # tabla_puntuacion = str.maketrans('', '', string.punctuation)
#     # texto_sin_puntuacion = texto.translate(tabla_puntuacion)
#     # if not conservar_espacios:
#     #     texto_sin_puntuacion = remover_espacios_extra(texto_sin_puntuacion)
#     # return texto_sin_puntuacion
#     pass # Implementación de ejemplo omitida por brevedad