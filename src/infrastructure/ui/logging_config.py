import logging
import logging.config
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.domain.interfaces import ILogger, IUserInterface

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
        self._logger.propagate = False

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def set_session_file(self, log_path: Path) -> None:
        """Base logger (CLI) tidak memerlukan file sesi terpisah."""
        pass

class ContextualLogger(ILogger):
    """
    Routes logs to both a specific UI session and a global base logger.
    Ensures that multi-user environments (like Gradio) don't mix logs.
    """
    def __init__(self, ui: IUserInterface, base_logger: ILogger):
        self.ui = ui
        self.base_logger = base_logger
        self._session_logger: Optional[logging.Logger] = None

    def set_session_file(self, log_path: Path):
        """Mengarahkan log sesi ke file fisik di folder State."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Gunakan logger unik per instance agar aman di lingkungan multi-user
        logger_name = f"Session_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        if not logger.handlers:
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
        
        self._session_logger = logger

    def _log_session(self, level: int, msg: str, *args, **kwargs):
        if self._session_logger:
            self._session_logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log_session(logging.DEBUG, msg, *args, **kwargs)
        self.base_logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        formatted_msg = msg % args if args else msg
        self.ui.show_info(formatted_msg, level="INFO")
        self._log_session(logging.INFO, msg, *args, **kwargs)
        self.base_logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log_session(logging.WARNING, msg, *args, **kwargs)
        self.base_logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        formatted_msg = msg % args if args else msg
        self.ui.show_info(formatted_msg, level="ERROR")
        self._log_session(logging.ERROR, msg, *args, **kwargs)
        self.base_logger.error(msg, *args, **kwargs)
