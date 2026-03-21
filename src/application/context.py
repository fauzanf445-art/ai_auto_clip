from dataclasses import dataclass
from typing import Optional

from src.domain.interfaces import IUserInterface, IProgressReporter

@dataclass
class SessionContext:
    """
    Menyimpan state yang spesifik untuk satu permintaan (request) atau sesi user.
    Objek ini akan dipassing dari Controller ke Service method.
    """
    ui: IUserInterface
    api_key: str
    progress_reporter: Optional[IProgressReporter] = None
