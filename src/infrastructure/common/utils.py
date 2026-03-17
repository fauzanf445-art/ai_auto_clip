import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
import shutil

from src.domain.exceptions import ExecutableNotFoundError

class JsonCache:
    """Utilitas untuk menangani cache data dalam format JSON."""
    @staticmethod
    def load(path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            logging.debug(f"♻️ Memuat dari cache: {path.name}")
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"⚠️ Cache korup atau tidak valid ({path.name}): {e}")
            return None

    @staticmethod
    def save(data: Any, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            logging.debug(f"💾 Disimpan ke cache: {path.name}")
            return True
        except Exception as e:
            logging.error(f"❌ Gagal menyimpan cache ({path.name}): {e}")
            return False

def sanitize_filename(name: str) -> str:
    """Membersihkan string agar aman digunakan sebagai nama file/folder."""
    # Hapus semua karakter yang bukan alfanumerik, spasi, strip, atau underscore
    raw_safe = re.sub(r'[^\w\s\-_]', '', name).strip()
    # Ganti beberapa spasi atau karakter whitespace lainnya menjadi satu spasi tunggal
    return re.sub(r'\s+', ' ', raw_safe)

def find_executable(name: str) -> str:
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
    logging.info(f"✅ Ditemukan executable '{name}' di: {path}")
    return path