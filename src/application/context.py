from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.domain.interfaces import IUserInterface, IProgressReporter, ILogger

@dataclass
class SessionContext:
    """
    Menyimpan state yang spesifik untuk satu permintaan (request) atau sesi user.
    Objek ini akan dipassing dari Controller ke Service method.
    """
    ui: IUserInterface
    api_key: str
    url: str
    logger: ILogger
    progress_reporter: Optional[IProgressReporter] = None
    work_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    state_dir: Optional[Path] = None
