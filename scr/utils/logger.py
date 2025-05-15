# utils/logger.py

import logging
from pathlib import Path

def setup_logger(name, ruta_log_file=None, level=logging.INFO):
    if ruta_log_file:
        # Crea directorio si no existe
        Path(ruta_log_file).parent.mkdir(parents=True, exist_ok=True)

    # Crea el logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # evita que los mensajes se dupliquen a root

    # Evita duplicar handlers si se llama varias veces
    if not logger.handlers:
        # Formato de los logs
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        # handler de archivo, sólo si ruta_log_file no es None
        if ruta_log_file:
            # Log a archivo
            fh = logging.FileHandler(ruta_log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # handler de consola
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
