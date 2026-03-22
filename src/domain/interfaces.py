from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any, Dict, Optional, Callable, Iterable, Tuple, Protocol, runtime_checkable
from .models import Clip, VideoSummary, TranscriptionWord, TranscriptionSegment, TrackResult

class IVideoProcessor(ABC):
    
    @property
    @abstractmethod
    def is_gpu_enabled(self) -> bool: ...

    @abstractmethod
    def render_final(self, video_path: str, audio_path: str, subtitle_path: Optional[str], output_path: str, fonts_dir: Optional[str] = None) -> None: ...

class ICacheManager(ABC):
    """Interface untuk mekanisme caching."""
    @abstractmethod
    def load(self, path: str) -> Optional[Any]: ...
    
    @abstractmethod
    def save(self, data: Any, path: str) -> None: ...

class ITranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscriptionSegment]: ...

class IContentAnalyzer(ABC):
    @abstractmethod
    def analyze_content(self, transcript: str, audio_path: str, prompt: str, api_key: str = "") -> VideoSummary: ...

class IFaceTracker(ABC):
    @abstractmethod
    def track_and_crop(self, input_path: str, output_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> TrackResult: ...

class IMediaDownloader(ABC):
    """Interface untuk mengunduh media dan metadata (yt-dlp)."""
    @abstractmethod
    def get_safe_title(self, url: str) -> str: ...
    
    @abstractmethod
    def download_audio(self, url: str, output_dir: str, filename_prefix: str) -> str: ...
    
    @abstractmethod
    def download_video_section(self, url: str, start: float, end: float, output_path: str) -> None: ...

    @abstractmethod
    def get_transcript(self, url: str, output_dir: str, filename_prefix: str) -> str: ...

class ICookieExtractor(ABC):
    """Interface khusus untuk mengekstrak cookies browser."""
    @abstractmethod
    def extract_cookies(self, browser: str, output_path: str) -> None: ...

class ISubtitleWriter(ABC):
    """Interface for writing subtitle files."""
    @abstractmethod
    def write_karaoke_subtitles(
        self, 
        transcription_data: List[TranscriptionSegment], 
        output_path: str, 
        chunk_size: int, 
        play_res_x: int, 
        play_res_y: int
    ) -> None: ...

class IProgressBar(ABC):
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
    @abstractmethod
    def sequence(self, iterable: Iterable, desc: str = "", unit: str = "it", total: Optional[int] = None) -> Iterable: ...
    
    @abstractmethod
    def manual(self, total: int, desc: str = "", unit: str = "it", leave: bool = True) -> IProgressBar: ...

class IFileDownloader(ABC):
    @abstractmethod
    def download(self, url: str, destination: str) -> None: ...

class ITextProcessor(ABC):
    @abstractmethod
    def extract_json(self, text: str) -> Dict[str, Any]: ...

class IRetryHandler(ABC):
    @abstractmethod
    def execute(self, func: Callable, *args, **kwargs) -> Any: ...

class ILogger(ABC):
    """Interface untuk abstraksi logging aplikasi."""
    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None: ...

    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None: ...

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

class ISystemHelper(ABC):
    """Interface untuk interaksi sistem operasi."""
    @abstractmethod
    def find_executable(self, name: str) -> str: ...
    @abstractmethod
    def prune_directory(self, directory: Path, max_files: int, max_size_mb: int, file_prefix: str = "", extensions: Tuple[str, ...] = ()) -> None: ...

@runtime_checkable
class IWorkspaceManager(Protocol):
    """Protocol untuk Workspace Manager (Context Manager)."""
    def __enter__(self) -> Tuple[str, Path]: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

@runtime_checkable
class IWorkspaceFactory(Protocol):
    """Protocol untuk Factory pembuatan Workspace."""
    def create(self, raw_name: str) -> IWorkspaceManager: ...

@runtime_checkable
class IProviderService(Protocol):
    """Protocol untuk Provider Service."""
    def get_safe_folder_name(self, url: str) -> Optional[str]: ...
    
    def get_transcript(self, url: str, temp_dir: str, filename_prefix: str ) -> Path: ...
    
    def prepare_media_for_analysis(self, url: str, work_dir: Path, filename_prefix: str) -> Path: ...
    
    def analyze_video(self, transcript: str, audio_path: str, prompt: str, cache_path: Optional[str] = None, api_key: str = "") -> VideoSummary: ...

@runtime_checkable
class IEditorService(Protocol):
    """Protocol untuk Editor Service."""
    def batch_create_clips(self, clips: List[Clip], source_url: str, output_dir: Path, progress_reporter: Optional[IProgressReporter] = None, cookies_path: Optional[str] = None) -> List[Path]: ...
    
    def track_subject(self, input_path: str, output_path: str, progress_reporter: Optional[IProgressReporter] = None) -> TrackResult: ...
    
    def batch_render(
        self,
        tracked_results: List[Tuple[Path, TrackResult]],
        work_dir: Path,
        output_dir: Path,
        progress_reporter: Optional[IProgressReporter] = None
    ) -> List[Path]: ...
    
    def prune_output_directory(self, output_dir: Path, max_files: int = 10, max_size_mb: int = 500): ...