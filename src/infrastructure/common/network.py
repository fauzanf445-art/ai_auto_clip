import urllib.request
from pathlib import Path

from src.domain.exceptions import MediaDownloadError
from src.domain.interfaces import IFileDownloader, ILogger

class UrllibDownloader(IFileDownloader):
    """Implementasi downloader menggunakan urllib."""
    def __init__(self, logger: ILogger):
        self.logger = logger
    
    def download(self, url: str, destination: str) -> None:
        dest_path = Path(destination)
        self.logger.info(f"⬇️  Mengunduh dari {url} ke {dest_path}...")
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest_path)
            self.logger.info(f"✅ Berhasil diunduh: {dest_path.name}")
        except Exception as e:
            raise MediaDownloadError(f"Gagal mengunduh file: {e}", original_exception=e)

class AssetManager:
    """
    Mengelola aset eksternal yang diperlukan aplikasi.
    Menggunakan Dependency Injection untuk downloader.
    """
    def __init__(self, downloader: IFileDownloader, logger: ILogger):
        self.downloader = downloader
        self.logger = logger

    def ensure_asset(self, url: str, target_path: Path, description: str = "Asset"):
        """Memastikan aset tersedia, mengunduh jika belum ada."""
        if not target_path.exists():
            self.logger.info(f"⬇️  {description} tidak ditemukan. Mengunduh ke: {target_path}...")
            try:
                self.downloader.download(url, str(target_path))
            except Exception as e:
                self.logger.error(f"❌ Gagal mengunduh {description}: {e}")
                raise