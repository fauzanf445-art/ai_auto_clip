import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

from src.domain.exceptions import ExecutableNotFoundError
from src.domain.interfaces import ISystemHelper, ILogger, IWorkspaceManager, IWorkspaceFactory

class SystemHelper(ISystemHelper):
    """Kelas utilitas untuk interaksi sistem operasi dan hardware."""
    def __init__(self, logger: ILogger):
        self.logger = logger
    
    def find_executable(self, name: str) -> str:
        """
        Mencari path absolut untuk sebuah executable di sistem.

        Raises:
            ExecutableNotFoundError: Jika executable tidak ditemukan di PATH.
        """
        path = shutil.which(name)
        if path is None:
            raise ExecutableNotFoundError(
                f"Executable '{name}' tidak ditemukan di PATH sistem. "
                f"Harap pastikan '{name}' sudah terinstall dan bisa diakses secara global. "
                "Lihat README.md untuk instruksi instalasi."
            )
        self.logger.debug(f"✅ Ditemukan executable '{name}' di: {path}")
        return path

    def prune_directory(self, directory: Path, max_files: int, max_size_mb: int, file_prefix: str = "", extensions: Tuple[str, ...] = ()) -> None:
        """
        Membersihkan file dalam direktori berdasarkan batas jumlah dan ukuran.
        Menghapus file terlama terlebih dahulu.
        """
        self.logger.info(f"Memeriksa folder untuk pemangkasan: {directory}")
        try:
            files_found = []
            # Gunakan os.walk untuk support recursive scanning jika diperlukan, atau ganti Path.glob
            for root, _, files in os.walk(directory):
                for file in files:
                    # Filter berdasarkan prefix dan extension
                    if file.lower().startswith(file_prefix.lower()) and file.lower().endswith(extensions):
                        file_path = Path(root) / file
                        try:
                            stat = file_path.stat()
                            # Simpan tuple (path, modified_time, size)
                            files_found.append((file_path, stat.st_mtime, stat.st_size))
                        except FileNotFoundError:
                            continue
            
            # Sort berdasarkan waktu (terlama di awal)
            files_found.sort(key=lambda x: x[1])

            # 1. Hapus jika melebihi jumlah file (Count Limit)
            while len(files_found) > max_files:
                to_remove = files_found.pop(0)
                try:
                    to_remove[0].unlink()
                    self.logger.info(f"🗑️ Pruned (Limit File): {to_remove[0].name}")
                except OSError as e:
                    self.logger.error(f"Gagal menghapus {to_remove[0]}: {e}")

            # 2. Hapus jika melebihi ukuran total (Size Limit)
            total_size = sum(f[2] for f in files_found)
            max_size_bytes = max_size_mb * 1024 * 1024

            while total_size > max_size_bytes and files_found:
                to_remove = files_found.pop(0)
                try:
                    to_remove[0].unlink()
                    total_size -= to_remove[2]
                    self.logger.info(f"🗑️ Pruned (Limit Size): {to_remove[0].name}")
                except OSError as e:
                    self.logger.error(f"Gagal menghapus {to_remove[0]}: {e}")
        
        except Exception as e:
            self.logger.error(f"Terjadi error saat memangkas folder: {e}")

class WorkspaceManager(IWorkspaceManager):
    """
    Context manager untuk mengelola lifecycle folder sementara.
    Otomatis membuat folder saat masuk dan membersihkannya saat keluar.
    """
    def __init__(self, base_dir: Path, raw_name: str, logger: ILogger, keep_on_exit: bool = False):
        self.base_dir = base_dir
        self.raw_name = raw_name
        self.logger = logger
        self.keep_on_exit = keep_on_exit
        self.work_dir: Optional[Path] = None

    def __enter__(self) -> Tuple[str, Path]:
        safe_name = self.raw_name
        self.work_dir = self.base_dir / self.raw_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return safe_name, self.work_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.work_dir and self.work_dir.exists(): 
            if not self.keep_on_exit:
                self.logger.info(f"🧹 Membersihkan workspace: {self.work_dir}")
                shutil.rmtree(self.work_dir, ignore_errors=True)
            else:
                self.logger.info(f"🛑 Debug Mode: Workspace tidak dibersihkan: {self.work_dir}")

class WorkspaceManagerFactory(IWorkspaceFactory):
    """
    Factory untuk membuat WorkspaceManager.
    Mencapsulasi dependensi statis (base_dir, logger) agar tidak perlu diteruskan manual oleh caller.
    """
    def __init__(self, base_dir: Path, logger: ILogger, keep_temp_dirs: bool = False):
        self.base_dir = base_dir
        self.logger = logger
        self.keep_temp_dirs = keep_temp_dirs
    
    def create(self, raw_name: str) -> IWorkspaceManager:
        return WorkspaceManager(self.base_dir, raw_name, self.logger, keep_on_exit=self.keep_temp_dirs)