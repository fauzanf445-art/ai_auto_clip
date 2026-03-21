import os
from pathlib import Path
from typing import Optional, Union

from src.domain.interfaces import ILogger, ICookieExtractor

class AuthService:
    """
    Layanan untuk menangani autentikasi dan kredensial (Cookies).
    """
    def __init__(self, cookie_extractor: ICookieExtractor, logger: ILogger):
        self.cookie_extractor = cookie_extractor
        self.logger = logger

    def extract_cookies_from_browser(self, target_path: Path) -> bool:
        supported_browsers = ["chrome", "firefox", "edge", "opera", "brave"]
        
        for browser in supported_browsers:
            try:
                self.logger.debug(f"Mencoba mengambil cookies dari browser: {browser}...")
                self.cookie_extractor.extract_cookies(browser, str(target_path))
                
                if target_path.exists() and target_path.stat().st_size > 0:
                    self.logger.info(f"✅ File cookies berhasil dibuat dari {browser}: {target_path}")
                    return True
            except Exception as e:
                self.logger.debug(f"Gagal mengambil cookies dari {browser}: {e}")
                continue
        
        return False

    def check_and_setup_cookies(self, cookies_path: Union[str, Path]) -> Optional[Path]:
        path_obj = Path(cookies_path)
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

        if os.getenv("SPACE_ID"):
            self.logger.warning("⚠️ Berjalan di lingkungan Cloud. Ekstraksi cookies browser dilewati.")
            return None

        if self.extract_cookies_from_browser(path_obj):
            return path_obj
        
        self.logger.warning("⚠️ Gagal mengekstrak cookies. YouTube mungkin memblokir akses (Sign-in Required).")
        return None