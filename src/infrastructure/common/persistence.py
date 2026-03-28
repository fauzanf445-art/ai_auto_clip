import json
from pathlib import Path
from typing import Any, Optional

from src.application.context import SessionContext
from src.domain.interfaces import IUtilsCacheManager, ILogger

class JsonFileCache(IUtilsCacheManager):
    """
    Implementasi ICacheManager menggunakan JSON file.
    Menggabungkan logika I/O langsung di sini untuk mengurangi kompleksitas.
    """
    def __init__(self, logger: ILogger):
        self.logger = logger

    def load(self, ctx: SessionContext, path: str) -> Optional[Any]:
        path_obj = Path(path)
        if not path_obj.exists():
            return None
        try:
            ctx.logger.debug(f"♻️ Memuat dari cache: {path_obj.name}")
            return json.loads(path_obj.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, Exception) as e:
            ctx.logger.warning(f"⚠️ Cache korup atau tidak valid ({path_obj.name}): {e}")
            return None

    def save(self, ctx: SessionContext, data: Any, path: str) -> None:
        path_obj = Path(path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            ctx.logger.debug(f"💾 Disimpan ke cache: {path_obj.name}")
        except Exception as e:
            # Jangan ditelan, lempar agar caller tahu (atau catch untuk warning)
            raise IOError(f"Gagal menyimpan cache ke {path_obj.name}: {e}")