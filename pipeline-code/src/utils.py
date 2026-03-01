import logging
import sys
import os

def setup_logger(name="pipeline", log_file="pipeline.log", level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    
    return logger

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
