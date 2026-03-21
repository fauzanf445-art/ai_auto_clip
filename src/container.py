import threading
from src.config import AppConfig

# Adapters
from src.infrastructure.ui.logging_config import TqdmLogger
from src.infrastructure.adapters.youtube_adapter import YouTubeAdapter
from src.infrastructure.adapters.ffmpeg_adapter import FFmpegAdapter
from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.infrastructure.adapters.whisper_adapter import WhisperAdapter
from src.infrastructure.adapters.mediapipe_adapter import MediaPipeAdapter
from src.infrastructure.adapters.subtitle_writer import AssSubtitleWriter
from src.infrastructure.common.filesystem import SystemHelper, WorkspaceManager, WorkspaceManagerFactory
from src.infrastructure.common.persistence import JsonFileCache
from src.infrastructure.common.network import UrllibDownloader, AssetManager
from src.infrastructure.common.text import RegexTextProcessor
from src.infrastructure.common.resilience import RetryHandler

from src.domain.exceptions import ExecutableNotFoundError, RateLimitError
from src.domain.interfaces import IProviderService, IEditorService, IWorkspaceFactory

# Services
from src.application.services.provider_service import ProviderService
from src.application.services.editor_service import EditorService
from src.application.services.auth_service import AuthService
from src.application.workflow import Workflow

class Container:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Implementasi Thread-Safe Singleton menggunakan Double-Checked Locking."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Container, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: AppConfig, verbose: bool = False):
        # Jika sudah diinisialisasi, cukup segarkan dependensi context dan return
        if self._initialized:
            return

        self.config = config
        self.verbose = verbose

        # 0. Initialize Logger (Singleton Scope within Container)
        # Pass the verbose flag to the logger
        self.logger = TqdmLogger(self.config.paths.LOG_FILE, verbose=self.verbose)

        # Initialize components in order of dependency
        self._init_infrastructure()
        self._init_adapters()
        self._init_services()
        self._init_orchestrator()
        
        # Tandai bahwa sistem berat sudah dimuat
        self._initialized = True

    def _init_infrastructure(self):
        """Inisialisasi komponen dasar sistem, cache, dan utilitas."""
        self.system = SystemHelper(self.logger)
        self.cache_manager = JsonFileCache(self.logger)
        self.file_downloader = UrllibDownloader(self.logger)
        self.text_processor = RegexTextProcessor()
        
        # 1. General Retry Handler (Default: Exponential Backoff)
        self.retry_handler = RetryHandler(self.logger)
        
        # 2. Transcript Retry Handler (Policy: Cool Down 45s, Constant Delay)
        # Hanya retry jika terkena RateLimitError
        self.transcript_retry_handler = RetryHandler(
            self.logger, max_attempts=2, initial_delay=45.0, backoff_factor=1.0, 
            retry_on=(RateLimitError,)
        )
        
        self.asset_manager = AssetManager(self.file_downloader, self.logger)
        
        self.asset_manager.ensure_asset(
            "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
            self.config.paths.FONTS_DIR / "Poppins-Bold.ttf",
            "Font Utama"
        )
        self.asset_manager.ensure_asset(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            self.config.paths.FACE_LANDMARKER_FILE,
            "Model MediaPipe"
        )
        
        # Workspace Factory
        self.workspace_factory = WorkspaceManagerFactory(self.config.paths.TEMP_DIR, self.logger, keep_temp_dirs=self.verbose)

    def _init_adapters(self):
        """Inisialisasi adapter eksternal dan pencarian executable."""
        # 1. System Binaries Lookup
        ffmpeg_path = self.system.find_executable("ffmpeg")
        ytdlp_path = self.system.find_executable("yt-dlp")
        ffprobe_path = self.system.find_executable("ffprobe")
        
        node_path = None
        try:
            node_path = self.system.find_executable("node")
        except ExecutableNotFoundError:
            pass

        # 2. Adapters Instantiation
        self.yt_adapter = YouTubeAdapter(
            yt_dlp_path=ytdlp_path,
            node_path=node_path,
            cookies_path=self.config.paths.COOKIE_FILE,
            logger=self.logger
        )
        
        self.ffmpeg_adapter = FFmpegAdapter(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            encoder_preference=self.config.ffmpeg_encoder_preference,
            logger=self.logger
        )
        
        self.gemini_adapter = GeminiAdapter(
            api_key="", # Singleton tidak menyimpan API Key user, akan dipassing saat runtime
            model_names=self.config.gemini_models,
            text_processor=self.text_processor,
            retry_handler=self.retry_handler,
            logger=self.logger
        )
        
        whisper_hw = WhisperAdapter.detect_hardware(self.logger)
        self.whisper_adapter = WhisperAdapter(
            **whisper_hw, 
            download_root=str(self.config.paths.WHISPER_MODELS_DIR),
            logger=self.logger
        )
        
        self.mp_adapter = MediaPipeAdapter(
            model_path=str(self.config.paths.FACE_LANDMARKER_FILE), 
            retry_handler=self.retry_handler,
            window_size=self.config.motion_window_size,
            process_every_n_frames=self.config.motion_process_every_n_frames,
            logger=self.logger
        )

        self.subtitle_writer = AssSubtitleWriter(config=self.config.subtitle, logger=self.logger)

    def _init_services(self):
        """Inisialisasi Application Services."""
        self.auth_service = AuthService(
            cookie_extractor=self.yt_adapter,
            logger=self.logger
        )

        self.provider_service = ProviderService(
            downloader=self.yt_adapter,
            processor=self.ffmpeg_adapter,
            analyzer=self.gemini_adapter,
            cache_manager=self.cache_manager,
            retry_handler=self.transcript_retry_handler,
            logger=self.logger
        )

        # Runtime check untuk memastikan ProviderService memenuhi Protocol IProviderService
        if not isinstance(self.provider_service, IProviderService):
            self.logger.warning("⚠️ ProviderService tidak memenuhi kontrak IProviderService secara runtime.")
            # Opsional: raise TypeError("ProviderService implementation violation")
        
        self.editor_service = EditorService(
            downloader=self.yt_adapter,
            processor=self.ffmpeg_adapter,
            tracker=self.mp_adapter,
            transcriber=self.whisper_adapter,
            writer=self.subtitle_writer,
            system_helper=self.system,
            fonts_dir=self.config.paths.FONTS_DIR,
            karaoke_chunk_size=self.config.karaoke_chunk_size,
            logger=self.logger
        )

        # Runtime check untuk memastikan EditorService memenuhi Protocol IEditorService
        if not isinstance(self.editor_service, IEditorService):
            self.logger.warning("⚠️ EditorService tidak memenuhi contract IEditorService secara runtime.")
            # Opsional: raise TypeError("EditorService implementation violation")

    def _init_orchestrator(self):
        """Inisialisasi Workflow Orchestrator."""
        # Runtime check untuk WorkspaceFactory
        if not isinstance(self.workspace_factory, IWorkspaceFactory):
             self.logger.warning("⚠️ WorkspaceFactory tidak memenuhi kontrak IWorkspaceFactory secara runtime.")

        self.orchestrator = Workflow(
            self.config, 
            provider=self.provider_service, 
            editor=self.editor_service,
            manager_factory=self.workspace_factory,
            logger=self.logger
        )