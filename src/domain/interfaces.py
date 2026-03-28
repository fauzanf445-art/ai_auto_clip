from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any, Dict, Optional, Callable, Iterable, Tuple, Protocol, runtime_checkable, TYPE_CHECKING
from .models import Clip, VideoSummary, TranscriptionWord, TranscriptionSegment, TrackResult, ProjectState

if TYPE_CHECKING:
    from src.application.context import SessionContext


@runtime_checkable
class IAppPaths(Protocol):
    """
    Interface fleksibel untuk manajemen path aplikasi.
    Menghilangkan enum/key statis agar konfigurasi folder bisa dikelola manual di config.py.
    """
    @property
    def temp_dir(self) -> Path: ...
    @property
    def output_dir(self) -> Path: ...
    @property
    def logs_dir(self) -> Path: ...
    @property
    def state_dir(self) -> Path: ...
    @property
    def models_dir(self) -> Path: ...
    @property
    def fonts_dir(self) -> Path: ...
    @property
    def whisper_models_dir(self) -> Path: ...
    @property
    def mediapipe_dir(self) -> Path: ...
    @property
    def ai_cache_dir(self) -> Path: ...
    @property
    def log_file(self) -> Path: ...
    @property
    def cookie_file(self) -> Path: ...
    @property
    def prompt_file(self) -> Path: ...
    @property
    def face_landmarker_file(self) -> Path: ...
    @property
    def env_file(self) -> Path: ...
    @property
    def raw_ai_filename(self) -> str: ...
    @property
    def summary_filename(self) -> str: ...
    @property
    def state_filename(self) -> str: ...
    @property
    def all_directories(self) -> List[Path]: ...

@runtime_checkable
class ISubtitleConfig(Protocol):
    """Kontrak konfigurasi untuk Subtitle Writer."""
    font_name: str
    font_size: int
    margin_v: int
    primary_color: str
    secondary_color: str
    outline_color: str
    back_color: str
    karaoke_chunk_size: int
    bold: int
    italic: int

@runtime_checkable
class IWhisperConfig(Protocol):
    """Kontrak konfigurasi untuk Whisper Adapter."""
    language: str
    initial_prompt: str
    use_batched_pipeline: bool
    log_prob_threshold: float
    compression_ratio_threshold: float

@runtime_checkable
class IAppConfig(Protocol):
    """Interface untuk akses konfigurasi aplikasi secara global."""
    @property
    def paths(self) -> IAppPaths: ...
    
    @property
    def ffmpeg_encoder_preference(self) -> Optional[str]: ...
    
    @property
    def gemini_models(self) -> List[str]: ...
    
    @property
    def subtitle(self) -> ISubtitleConfig: ...

    @property
    def whisper(self) -> IWhisperConfig: ...

# --- Workspace & Lifecycle Interfaces ---

@runtime_checkable
class IWorkspaceManager(Protocol):
    """Context Manager untuk siklus hidup folder kerja sementara."""
    def __enter__(self) -> Tuple[str, Path]: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

@runtime_checkable
class IWorkspaceFactory(Protocol):
    """Factory untuk pembuatan instance WorkspaceManager."""
    def create(self, ctx: "SessionContext", raw_name: str) -> IWorkspaceManager: ...

# --- Domain Service Interfaces ---

@runtime_checkable
class IManagerService(Protocol):
    """Layanan untuk memastikan kesiapan infrastruktur (folder & aset)."""
    def ensure_system_integrity(self) -> None: ...

@runtime_checkable
class IProviderService(Protocol):
    """Layanan untuk penyediaan data mentah (Video, Audio, Transkrip)."""
    def get_safe_folder_name(self, ctx: "SessionContext", url: str) -> Optional[str]: ...
    def get_prompt_for_analysis(self, ctx: "SessionContext") -> str: ...
    def warmup_ai(self, ctx: "SessionContext") -> None: ...
    def close_ai(self, ctx: "SessionContext") -> None: ...
    def get_audio_for_analysis(self, ctx: "SessionContext", url: str, temp_dir: Path, filename_prefix: str) -> Path: ...
    def analyze_video(self, ctx: "SessionContext", url: str, temp_dir: Path, filename_prefix: str, cache_path: str, audio_path: Optional[str] = None) -> VideoSummary: ...
    def load_project_state(self, ctx: "SessionContext", work_dir: str) -> ProjectState: ...
    def save_project_state(self, ctx: "SessionContext", work_dir: str, state: ProjectState) -> None: ...

@runtime_checkable
class IEditorService(Protocol):
    """Layanan untuk pemrosesan video dan rendering."""
    def warmup_ai(self, ctx: "SessionContext") -> None: ...
    def close_ai(self, ctx: "SessionContext") -> None: ...
    def batch_create_clips(self, ctx: "SessionContext", clips: List[Clip], source_url: str, output_dir: Path, cookies_path: Optional[str] = None) -> List[Path]: ...
    def track_subject(self, ctx: "SessionContext", input_path: str, output_path: str) -> TrackResult: ...
    def batch_render(self, ctx: "SessionContext", tracked_results: List[Tuple[Path, TrackResult]], clips: List[Clip], work_dir: Path, output_dir: Path) -> List[Path]: ...

# --- Domain Adapters Interfaces ---

class IFfmpegAdapter(ABC):
    """Interface untuk ffmpeg adapter"""
    @abstractmethod
    def is_gpu_enabled(self, ctx: "SessionContext") -> bool: ...

    @abstractmethod
    def get_video_duration(self, ctx: "SessionContext", path: str) -> Optional[float]: ...

    @abstractmethod
    def render_final(self, ctx: "SessionContext", video_path: str, audio_path: str, subtitle_path: Optional[str], output_path: str, fonts_dir: Optional[str] = None) -> None: ...

class IWhisperAdapter(ABC):
    """Interface untuk adapter Whisper."""
    @abstractmethod
    def ensure_model(self, ctx: "SessionContext") -> None: ...

    @abstractmethod
    def close(self, ctx: "SessionContext") -> None: ...

    @abstractmethod
    def transcribe(
        self, 
        ctx: "SessionContext",
        audio_path: str, 
        initial_prompt: Optional[str] = None,
        clip_timestamps: Optional[List[float]] = None
    ) -> Iterable[TranscriptionSegment]: ...

class IGeminiAdapter(ABC):
    """Interface untuk adapter AI Gemini"""
    @abstractmethod
    def upload_file(self, ctx: "SessionContext", file_path: str) -> Any: ...

    @abstractmethod
    def generate_content(self, ctx: "SessionContext", prompt: str, file_obj: Any = None, response_schema: Any = None) -> Any: ...

    @abstractmethod
    def delete_file(self, ctx: "SessionContext", file_name: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

class IMediapipeAdapter(ABC):
    """Interface untuk adapter Mediapipe."""
    @abstractmethod
    def ensure_model(self, ctx: "SessionContext") -> None: ...

    @abstractmethod
    def close(self, ctx: "SessionContext") -> None: ...

    @abstractmethod
    def track_and_crop(self, ctx: "SessionContext", input_path: str, output_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> TrackResult: ...

class IYoutubeAdapter(ABC):
    """Interface untuk mengunduh media dan metadata (yt-dlp)."""
    @abstractmethod
    def get_safe_title(self, ctx: "SessionContext", url: str) -> str: ...
    
    @abstractmethod
    def download_audio(self, ctx: "SessionContext", url: str, output_dir: str, filename_prefix: str) -> str: ...

    @abstractmethod
    def download_video_section(self, ctx: "SessionContext", url: str, start: float, end: float, output_path: str) -> None: ...

class ISubtitleWriter(ABC):
    """Interface for writing subtitle files."""
    @abstractmethod
    def write_ass_sub_style(self, ctx: "SessionContext", transcription_data: List[TranscriptionSegment], output_path: str, play_res_x: int, play_res_y: int ) -> None: ...

# ---Utility Interfaces ---

class ILogger(ABC):
    """Abstraksi logging agar tidak tergantung pada satu library tertentu."""
    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None: ...
    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None: ...
    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None: ...
    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def set_session_file(self, log_path: Path) -> None: ...

class IFileDownloader(ABC):
    """Interface untuk pengunduhan file mentah/aset."""
    @abstractmethod
    def download(self, ctx: Optional["SessionContext"], url: str, dest_path: Path, description: str) -> None: ...

class ISystemHelper(ABC):
    """Utilitas sistem operasi seperti pencarian executable dan pembersihan disk."""
    @abstractmethod
    def find_executable(self, name: str) -> str: ...

class IUtilsCacheManager(ABC):
    """Interface untuk mekanisme caching json."""
    @abstractmethod
    def load(self, ctx: "SessionContext", path: str) -> Optional[Any]: ...
    
    @abstractmethod
    def save(self, ctx: "SessionContext", data: Any, path: str) -> None: ...

class ITextProcessor(ABC):
    """Interface untuk pemrosesan teks."""
    @abstractmethod
    def extract_json(self, text: str) -> Dict[str, Any]: ...

class IProgressBar(ABC):
    """Interface Untuk UI Progress Bar"""
    @abstractmethod
    def update(self, n: int = 1): ...
    @abstractmethod
    def close(self): ...

    @property
    @abstractmethod
    def n(self) -> int: ...
    
    @property
    @abstractmethod
    def total(self) -> int: ...
    
    @total.setter
    @abstractmethod
    def total(self, value: int): ...

    @property
    @abstractmethod
    def disable(self) -> bool: ...

class IProgressReporter(ABC):
    """Interface Untuk Progress Reporter"""
    @abstractmethod
    def sequence(self, iterable: Iterable, desc: str = "", unit: str = "it", total: Optional[int] = None) -> Iterable: ...
    
    @abstractmethod
    def manual(self, total: int, desc: str = "", unit: str = "it", leave: bool = True) -> IProgressBar: ...

    @abstractmethod
    def set_logger(self, logger: "ILogger") -> None: ...

class IRetryHandler(ABC):
    """Interface untuk mekanisme retry handler(internet)"""
    @abstractmethod
    def execute(self, ctx: "SessionContext", func: Callable, *args, **kwargs) -> Any: ...

class IUserInterface(ABC):
    """
    Interface untuk interaksi pengguna (UI).
    Murni lapisan presentasi: Menerima input mentah dan menampilkan output.
    Tidak boleh ada logika bisnis/validasi di sini.
    """
    @abstractmethod
    def print_banner(self) -> None: ...

    @abstractmethod
    def get_input(self, prompt: str) -> str: ...
    
    @abstractmethod
    def get_secure_input(self, prompt: str) -> str: ...
    
    @abstractmethod
    def show_info(self, msg: str, level: str = "INFO") -> None: ...
    
    @abstractmethod
    def show_error(self, msg: str) -> None: ...

    @property
    @abstractmethod
    def log_output(self) -> str: ...

    @abstractmethod
    def log(self, msg: str) -> None: ...

    @abstractmethod
    def show_step(self, msg: str) -> None: ...

    @abstractmethod
    def create_demo(self, process_fn: Callable, cleanup_fn: Callable, default_api_key: str = "") -> Any: ...
