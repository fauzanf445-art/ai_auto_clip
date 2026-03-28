import shutil
from pathlib import Path
from typing import Tuple, Optional

from src.application.context import SessionContext
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

class WorkspaceManager(IWorkspaceManager):
    """
    Context manager untuk mengelola lifecycle folder sementara.
    Otomatis membuat folder saat masuk dan membersihkannya saat keluar.
    """
    def __init__(self, base_dir: Path, raw_name: str, ctx: SessionContext, clean_on_exit: bool = False):
        self.base_dir = base_dir
        self.raw_name = raw_name
        self.ctx = ctx
        self.clean_on_exit = clean_on_exit
        self.work_dir: Optional[Path] = None

    def __enter__(self) -> Tuple[str, Path]:
        safe_name = self.raw_name
        self.work_dir = self.base_dir / self.raw_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return safe_name, self.work_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.work_dir and self.work_dir.exists(): 
            if self.clean_on_exit:
                self.ctx.logger.info(f"🧹 Membersihkan workspace: {self.work_dir}")
                shutil.rmtree(self.work_dir, ignore_errors=True)
            else:
                self.ctx.logger.info(f"🛑 Mode Persisten: Workspace tidak dibersihkan: {self.work_dir}")

class WorkspaceManagerFactory(IWorkspaceFactory):
    """
    Factory untuk membuat WorkspaceManager.
    Mencapsulasi dependensi statis (base_dir, logger) agar tidak perlu diteruskan manual oleh caller.
    """
    def __init__(self, base_dir: Path, logger: ILogger, clean_on_exit: bool = False):
        self.base_dir = base_dir
        self.logger = logger
        self.temp_dirs = clean_on_exit
    
    def create(self, ctx: SessionContext, raw_name: str) -> IWorkspaceManager:
        return WorkspaceManager(self.base_dir, raw_name, ctx, clean_on_exit=self.temp_dirs)