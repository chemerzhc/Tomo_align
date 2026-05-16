import os
from loguru import logger


def setup_logger(output_dir):

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "pipeline.log")

    logger.remove()

    logger.add(
        log_path,
        rotation="100 MB",
        retention=5,
        level="INFO",
        enqueue=True
    )

    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO"
    )

    return logger