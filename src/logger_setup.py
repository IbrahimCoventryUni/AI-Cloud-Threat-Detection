import logging
import os

def setup_logger(log_file='threat_detection.log'):
    # Create a logger
    logger = logging.getLogger("ThreatDetection")
    logger.setLevel(logging.INFO)

    # Ensure there is no duplicate handler if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler to log to a file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create console handler to log to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and add it to both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    # Test the logger
    logger = setup_logger()
    logger.info("Logger is set up and working.")
