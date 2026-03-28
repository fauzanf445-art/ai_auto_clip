import urllib.request
from pathlib import Path
from typing import Optional

from src.application.context import SessionContext
from src.domain.exceptions import MediaDownloadError
from src.domain.interfaces import IFileDownloader, ILogger

class UrllibDownloader(IFileDownloader):
    """Implementasi downloader menggunakan urllib."""
    def __init__(self, logger: ILogger):
        self.logger = logger
    
    def download(self, ctx: Optional[SessionContext], url: str, dest_path: Path, description : str ) -> None:
        active_logger = ctx.logger if ctx else self.logger

        if not dest_path.exists():
            active_logger.info(f"⬇️ {description} Mengunduh dari {url} ke {dest_path}...")
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, dest_path)
                active_logger.info(f"✅ Berhasil diunduh: {dest_path.name}")
            except Exception as e:
                raise MediaDownloadError(f"Gagal mengunduh file: {e}", original_exception=e)
