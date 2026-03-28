from typing import Optional

# Konfigurasi
from src.infrastructure.config import AppConfig
from src.application.workflow import Workflow
from src.infrastructure.ui.logging_config import TqdmLogger

# Adapters
from src.infrastructure.adapters.youtube_adapter import YouTubeAdapter
from src.infrastructure.adapters.ffmpeg_adapter import FFmpegAdapter
from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.infrastructure.adapters.whisper_adapter import WhisperAdapter
from src.infrastructure.adapters.mediapipe_adapter import MediaPipeAdapter
from src.infrastructure.adapters.subtitle_writer import AssSubtitleWriter

# Infrastruktur
from src.infrastructure.common.filesystem import SystemHelper, WorkspaceManagerFactory
from src.infrastructure.common.persistence import JsonFileCache
from src.infrastructure.common.network import UrllibDownloader
from src.infrastructure.common.text import RegexTextProcessor

from src.domain.exceptions import ExecutableNotFoundError

# Services
from src.application.services.provider_service import ProviderService
from src.application.services.editor_service import EditorService
from src.application.services.auth_service import AuthService
from src.application.services.manager_service import ManagerService

class Container:
    """
    Composition Root yang menangani Dependency Injection.
    Sekarang mencakup ManagerService yang menangani integritas sistem secara mandiri.
    """
    def __init__(self, config: Optional[AppConfig] = None, clean_temp: bool = False, verbose: bool = False):
        self.config = config or AppConfig()
        self.clean_temp = clean_temp
        
        # Inisialisasi Logger secara internal berdasarkan config
        self.logger = TqdmLogger(self.config.paths.log_file, verbose=verbose)

        self._init_infrastructure()
        self._setup_auth() 
        self._init_adapters()
        self._init_services()

        # Bootstrap: Pastikan integritas sistem (folder & download aset) sebelum orkestrator berjalan
        self.manager_service.ensure_system_integrity()

        self._init_orchestrator()

    def _init_infrastructure(self):
        """Inisialisasi komponen dasar sistem, cache, dan utilitas."""
        self.system = SystemHelper(self.logger)
        self.cache_manager = JsonFileCache(self.logger)
        self.file_downloader = UrllibDownloader(self.logger)
        self.text_processor = RegexTextProcessor()
        
        # Tetap menggunakan Factory untuk manajemen lifecycle workspace yang bersih
        self.workspace_factory = WorkspaceManagerFactory(
            self.config.paths.temp_dir, 
            self.logger, 
            clean_on_exit=self.clean_temp
        )

    def _setup_auth(self):
        """Inisialisasi Auth Service dan setup kredensial awal."""
        self.auth_service = AuthService(logger=self.logger)
        self.auth_service.check_and_setup_cookies(self.config.paths.cookie_file)

    def _init_adapters(self):
        """Inisialisasi adapter eksternal dan pencarian executable."""
        ffmpeg_path = self.system.find_executable("ffmpeg")
        ytdlp_path = self.system.find_executable("yt-dlp")
        ffprobe_path = self.system.find_executable("ffprobe")

        # Node bersifat opsional untuk beberapa ekstraktor yt-dlp
        try:
            node_path = self.system.find_executable("node")
        except ExecutableNotFoundError:
            node_path = None

        self.yt_adapter = YouTubeAdapter(
            yt_dlp_path=ytdlp_path,
            logger=self.logger,
            node_path=node_path,
            cookies_path=self.config.paths.cookie_file,
        )
        
        self.ffmpeg_adapter = FFmpegAdapter(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            encoder_preference=self.config.ffmpeg_encoder_preference,
            logger=self.logger
        )
        
        self.gemini_adapter = GeminiAdapter(
            model_names=self.config.gemini_models,
            logger=self.logger
        )
        
        whisper_hw = WhisperAdapter.detect_hardware(self.logger)
        self.whisper_adapter = WhisperAdapter(
            config=self.config.whisper,
            **whisper_hw, 
            download_root=str(self.config.paths.whisper_models_dir),
            logger=self.logger
        )
        
        self.mp_adapter = MediaPipeAdapter(
            model_path=str(self.config.paths.face_landmarker_file), 
            logger=self.logger
        )

        self.subtitle_writer = AssSubtitleWriter(
            config=self.config.subtitle, 
            logger=self.logger
        )

    def _init_services(self):
        """Inisialisasi Application Services."""
        # ManagerService sekarang mengambil peran Bootstrap
        self.manager_service = ManagerService(
            config=self.config,
            utils_download=self.file_downloader,
            logger=self.logger
        )

        self.provider_service = ProviderService(
            downloader=self.yt_adapter,
            processor=self.ffmpeg_adapter,
            analyzer=self.gemini_adapter,
            transcriber=self.whisper_adapter,
            cache_manager=self.cache_manager,
            prompt_path=self.config.paths.prompt_file,
            logger=self.logger,
            ai_cache_dir=self.config.paths.ai_cache_dir,
            raw_ai_filename=self.config.paths.raw_ai_filename,
            summary_filename=self.config.paths.summary_filename,
            state_filename=self.config.paths.state_filename
        )
        
        self.editor_service = EditorService(
            config=self.config,
            downloader=self.yt_adapter,
            processor=self.ffmpeg_adapter,
            tracker=self.mp_adapter,
            writer=self.subtitle_writer,
            fonts_dir=self.config.paths.fonts_dir,
            logger=self.logger
        )

    def _init_orchestrator(self):
        """Inisialisasi Workflow Orchestrator."""
        self.orchestrator = Workflow(
            config=self.config, 
            provider=self.provider_service, 
            editor=self.editor_service,
            manager_factory=self.workspace_factory,
            logger=self.logger
        )

    @property
    def workflow(self) -> Workflow:
        """Mengembalikan instance Workflow utama."""
        return self.orchestrator