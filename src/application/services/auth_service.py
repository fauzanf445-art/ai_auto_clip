import os
from pathlib import Path
from typing import Optional, Union

class AuthService:
    """
    Layanan untuk menangani autentikasi dan kredensial (Cookies).
    """
    def __init__(self, logger):
        self.logger = logger

    def check_and_setup_cookies(self, cookies_path: Path) -> Optional[Path]:
        path_obj = cookies_path
        if path_obj.exists() and path_obj.stat().st_size > 0:
            self.logger.info(f"✅ File cookies ditemukan di: {path_obj}")
            return path_obj

        if env_cookies := os.getenv("YOUTUBE_COOKIES"):
            try:
                self.logger.info("🍪 Menemukan cookies dari Environment Variable. Menyimpan ke file...")
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.write_text(env_cookies, encoding='utf-8')
                return path_obj
            except Exception as e:
                self.logger.error(f"Gagal menyimpan cookies dari Env: {e}")

        self.logger.warning("⚠️ File cookies tidak ditemukan. YouTube mungkin memblokir akses (Sign-in Required).")
        return None