import logging
import logging.config
from pathlib import Path
from tqdm import tqdm

from src.domain.interfaces import ILogger

class TqdmLoggingHandler(logging.Handler):
    """
    Custom Logging Handler yang menggunakan tqdm.write() 
    agar output log tidak merusak tampilan progress bar yang sedang berjalan.
    """
    def emit(self, record: logging.LogRecord):
        try:
            # Ensure the message is a string before passing it to tqdm.write
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

class TqdmLogger(ILogger):
    """
    Implementasi ILogger yang membungkus konfigurasi logging standar.
    """
    def __init__(self, log_file: Path, verbose: bool = False):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        console_level = logging.DEBUG if verbose else logging.INFO

        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'file_format': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'console_format': {
                    'format': '%(asctime)s - %(levelname)s - %(message)s',
                    'datefmt': '%H:%M:%S'
                },
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'file_format',
                    'filename': str(log_file),
                    'maxBytes': 5*1024*1024,
                    'backupCount': 3,
                    'encoding': 'utf-8',
                    'level': logging.DEBUG,
                },
                'console': {
                    'class': 'src.infrastructure.ui.logging_config.TqdmLoggingHandler',
                    'formatter': 'console_format',
                    'level': console_level,
                },
            },
            'root': {
                'handlers': ['file', 'console'],
                'level': logging.DEBUG,
            },
            'loggers': {
                "googleapiclient": {"level": logging.WARNING, "propagate": False},
                "urllib3": {"level": logging.WARNING, "propagate": False},
                "absl": {"level": logging.WARNING, "propagate": False},
                "yt_dlp": {"level": logging.WARNING, "propagate": False},
                "httpx": {"level": logging.WARNING, "propagate": False},
                "httpcore": {"level": logging.WARNING, "propagate": False},
                "faster_whisper": {"level": logging.WARNING, "propagate": False},
                "google_genai": {"level": logging.WARNING, "propagate": False},
                "google.genai": {"level": logging.WARNING, "propagate": False},
                "mediapipe": {"level": logging.WARNING, "propagate": False},
            }
        }

        logging.config.dictConfig(logging_config)
        self._logger = logging.getLogger("HSUAIClip")

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)
