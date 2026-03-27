from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any, Dict, Optional, Callable, Iterable, Tuple, Protocol, runtime_checkable
from .models import Clip, VideoSummary, TranscriptionWord, TranscriptionSegment, TrackResult, ProjectState


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
    def ai_cache_dir(self) -> Path: ...
    @property
    def log_file(self) -> Path: ...
    @property
    def cookie_file(self) -> Path: ...
    @property
    def prompt_file(self) -> Path: ...
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
    def motion_window_size(self) -> int: ...
    
    @property
    def motion_process_every_n_frames(self) -> int: ...
    
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
    def create(self, raw_name: str) -> IWorkspaceManager: ...

# --- Domain Service Interfaces ---

@runtime_checkable
class IManagerService(Protocol):
    """Layanan untuk memastikan kesiapan infrastruktur (folder & aset)."""
    def ensure_system_integrity(self) -> None: ...

@runtime_checkable
class IProviderService(Protocol):
    """Layanan untuk penyediaan data mentah (Video, Audio, Transkrip)."""
    def get_safe_folder_name(self, url: str) -> Optional[str]: ...
    def get_prompt_for_analysis(self) -> str: ...
    def warmup_ai(self) -> None: ...
    def get_audio_for_analysis(self, url: str, temp_dir: Path, filename_prefix: str) -> Path: ...
    def analyze_video(self, url: str, temp_dir: Path, filename_prefix: str, cache_path: str, api_key: str = "", audio_path: Optional[str] = None) -> VideoSummary: ...
    def load_project_state(self, work_dir: str) -> ProjectState: ...
    def save_project_state(self, work_dir: str, state: ProjectState) -> None: ...

@runtime_checkable
class IEditorService(Protocol):
    """Layanan untuk pemrosesan video dan rendering."""
    def batch_create_clips(self, clips: List[Clip], source_url: str, output_dir: Path, progress_reporter: Optional[Any] = None) -> List[Path]: ...
    def track_subject(self, input_path: str, output_path: str, progress_reporter: Optional[Any] = None) -> TrackResult: ...
    def batch_render(self, tracked_results: List[Tuple[Path, TrackResult]], clips: List[Clip], work_dir: Path, output_dir: Path, progress_reporter: Optional[Any] = None) -> List[Path]: ...
    def prune_output_directory(self, output_dir: Path, max_files: int = 10, max_size_mb: int = 500): ...

# --- Domain Adapters Interfaces ---

class IFfmpegAdapter(ABC):
    """Interface untuk ffmpeg adapter"""
    @abstractmethod
    def is_gpu_enabled(self) -> bool: ...

    @abstractmethod
    def get_video_duration(self, path: str) -> Optional[float]: ...

    @abstractmethod
    def render_final(self, video_path: str, audio_path: str, subtitle_path: Optional[str], output_path: str, fonts_dir: Optional[str] = None) -> None: ...

class IWhisperAdapter(ABC):
    """Interface untuk adapter Whisper."""
    @abstractmethod
    def ensure_model(self) -> None: ...

    @abstractmethod
    def transcribe(
        self, 
        audio_path: str, 
        initial_prompt: Optional[str] = None,
        clip_timestamps: Optional[List[float]] = None
    ) -> Iterable[TranscriptionSegment]: ...

class IGeminiAdapter(ABC):
    """Interface untuk adapter AI Gemini"""
    @abstractmethod
    def upload_file(self, file_path: str, api_key: str = "") -> Any: ...

    @abstractmethod
    def generate_content(self, prompt: str, file_obj: Any = None, api_key: str = "", response_schema: Any = None) -> Any: ...

    @abstractmethod
    def delete_file(self, file_name: str, api_key: str = "") -> None: ...

    @abstractmethod
    def close(self) -> None: ...

class IMediapipeAdater(ABC):
    """Interface untuk adapter Mediapipe."""
    @abstractmethod
    def track_and_crop(self, input_path: str, output_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> TrackResult: ...

class IYoutubeAdapter(ABC):
    """Interface untuk mengunduh media dan metadata (yt-dlp)."""
    @abstractmethod
    def get_safe_title(self, url: str) -> str: ...
    
    @abstractmethod
    def download_audio(self, url: str, output_dir: str, filename_prefix: str) -> str: ...

    @abstractmethod
    def download_video_section(self, url: str, start: float, end: float, output_path: str) -> None: ...

class ISubtitleWriter(ABC):
    """Interface for writing subtitle files."""
    @abstractmethod
    def write_ass_sub_style(self, transcription_data: List[TranscriptionSegment], output_path: str, play_res_x: int, play_res_y: int ) -> None: ...

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

class IFileDownloader(ABC):
    """Interface untuk pengunduhan file mentah/aset."""
    @abstractmethod
    def download(self, url: str, dest_path: Path, description: str) -> None: ...

class ISystemHelper(ABC):
    """Utilitas sistem operasi seperti pencarian executable dan pembersihan disk."""
    @abstractmethod
    def find_executable(self, name: str) -> str: ...
    @abstractmethod
    def prune_directory(self, directory: Path, max_files: int, max_size_mb: int, file_prefix: str = "", extensions: Tuple[str, ...] = ()) -> None: ...

class IUtilsCacheManager(ABC):
    """Interface untuk mekanisme caching json."""
    @abstractmethod
    def load(self, path: str) -> Optional[Any]: ...
    
    @abstractmethod
    def save(self, data: Any, path: str) -> None: ...

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

class IRetryHandler(ABC):
    """Interface untuk mekanisme retry handler(internet)"""
    @abstractmethod
    def execute(self, func: Callable, *args, **kwargs) -> Any: ...

class IUserInterface(ABC):
    """
    Interface untuk interaksi pengguna (UI).
    Murni lapisan presentasi: Menerima input mentah dan menampilkan output.
    Tidak boleh ada logika bisnis/validasi di sini.
    """
    @abstractmethod
    def get_input(self, prompt: str) -> str: ...
    
    @abstractmethod
    def get_secure_input(self, prompt: str) -> str: ...
    
    @abstractmethod
    def show_info(self, msg: str) -> None: ...
    
    @abstractmethod
    def show_error(self, msg: str) -> None: ...
