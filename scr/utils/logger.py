#scr/utils/logger.py
"""
Módulo para la configuración de loggers (registradores de eventos).

Este módulo proporciona una función para configurar y obtener una instancia de logger
con handlers para consola y, opcionalmente, para archivo.
"""
import logging
from pathlib import Path
from typing import Union

def setup_logger(
    name: str, 
    ruta_log_file: Union[str, Path, None] = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configura y devuelve un logger con el nombre especificado.

    Si se proporciona `ruta_log_file`, se añade un FileHandler.
    Siempre se añade un StreamHandler para la salida por consola.
    Evita la duplicación de handlers si se llama múltiples veces con el mismo nombre.

    Args:
        name (str): El nombre del logger. Usualmente `__name__` o `Clase.__name__`.
        ruta_log_file (Union[str, Path, None], optional): Ruta al archivo de log. 
            Si es None, no se configura log a archivo. Defaults to None.
        level (int, optional): Nivel de logging (ej. `logging.INFO`, `logging.DEBUG`). 
            Defaults to logging.INFO.

    Returns:
        logging.Logger: La instancia del logger configurado.
    
    Raises:
        # Podría lanzar excepciones de I/O si la creación del FileHandler falla,
        # pero la función actual no las maneja explícitamente.
        # Por ahora, no documentaremos explícitamente 'Raises' a menos que añadamos manejo.
    """
    log_path = None # Inicializar log_path
    if ruta_log_file:
        log_path = Path(ruta_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # Evita que los mensajes se dupliquen al logger raíz si este también tiene handlers

    # Evita añadir handlers duplicados si el logger ya fue configurado
    if not logger.handlers:
        # Formato estándar para los mensajes de log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para archivo, solo si se especificó una ruta y log_path está definido
        if log_path: # Asegurarse que log_path no sea None
            fh = logging.FileHandler(str(log_path), encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Handler para consola
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
