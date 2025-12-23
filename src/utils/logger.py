import logging
import os
from datetime import datetime

# Log dosyasının adı: logs/train_2023-10-27_15-30.log gibi olacak
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Terminale de yazsın (Hem dosyaya hem ekrana)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

def get_logger(name):
    return logging.getLogger(name)

if __name__ == "__main__":
    # Test edelim
    logger = get_logger("TestLogger")
    logger.info("Bu bir bilgi mesajıdır.")
    logger.warning("Bu bir uyarıdır!")
    logger.error("Eyvah hata oluştu.")