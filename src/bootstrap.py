from src.config import AppConfig

class Bootstrap:
    """Menangani inisialisasi lingkungan dan I/O awal aplikasi."""

    @staticmethod
    def setup_directories(config: AppConfig):
        """Membuat struktur direktori yang diperlukan."""
        paths_to_create = [
            config.paths.TEMP_DIR, 
            config.paths.OUTPUT_DIR, 
            config.paths.MODELS_DIR,
            config.paths.FILES_DIR, 
            config.paths.FONTS_DIR,
            config.paths.LOGS_DIR, 
            config.paths.WHISPER_MODELS_DIR, 
            config.paths.MEDIAPIPE_DIR
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
