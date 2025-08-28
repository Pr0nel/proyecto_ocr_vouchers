#scr/utils/config_loader.py
"""
Módulo encargado de cargar la configuración desde un archivo YAML.
"""
import yaml

def load_config(path: str) -> dict:
    """
    Carga un archivo de configuración YAML desde la ruta especificada.

    Args:
        path (str): La ruta al archivo YAML.

    Returns:
        dict: Un diccionario con la configuración cargada.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra en la ruta especificada.
        yaml.YAMLError: Si ocurre un error durante el parseo del archivo YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
