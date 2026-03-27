from src.domain.interfaces import ILogger, IFileDownloader, IManagerService

class ManagerService(IManagerService):
    """
    Layanan tunggal untuk memastikan kesiapan sistem (Provisioning).
    Menggabungkan logika pembuatan direktori dan pengunduhan aset.
    """
    def __init__(self, config , utils_download: IFileDownloader, logger: ILogger):
        self.config = config
        self.utils = utils_download
        self.logger = logger

    def _setup_directories(self) -> None:
        """
        Internal: Membuat semua struktur direktori secara otomatis.
        Menggunakan properti 'all_directories' yang baru untuk fleksibilitas total.
        """
        for path in self.config.paths.all_directories:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"📁 Membuat direktori: {path}")

    def ensure_system_integrity(self) -> None:
        """
        EntryPoint Utama: Menyiapkan folder dan mengunduh aset yang diperlukan.
        Menggunakan metode .get() untuk mengambil path berdasarkan key string.
        """
        self.logger.info("🛠️ Memeriksa integritas sistem dan aset...")
        
        # 1. Jalankan pembuatan semua folder yang terdaftar di config
        self._setup_directories()

        # 2. Download Font
        fonts_dir = self.config.paths.fonts_dir
        self.utils.download(
            "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
            fonts_dir / "Poppins-Bold.ttf",
            "Font Utama (Poppins)"
        )

        # 3. Download Model MediaPipe
        face_model_path = self.config.paths.face_landmarker_file
        self.utils.download(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            face_model_path,
            "Model MediaPipe (Face Landmarker)"
        )
        
        self.logger.info("✅ Sistem siap digunakan.")