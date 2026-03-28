from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from dotenv import load_dotenv

from src.domain.interfaces import IAppPaths

@dataclass
class AppPaths(IAppPaths):
    """
    Implementasi IAppPaths yang fleksibel menggunakan dictionary internal.
    Anda bisa mengubah nilai path di sini tanpa merusak kontrak interface.
    """
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())

    @property
    def temp_dir(self) -> Path: return self.base_dir / "Temp"

    @property
    def output_dir(self) -> Path: return self.base_dir / "Output"

    @property
    def models_dir(self) -> Path: return self.base_dir / "resources" / "models"

    @property
    def fonts_dir(self) -> Path: return self.base_dir / "resources" / "fonts"

    @property
    def logs_dir(self) -> Path: return self.base_dir / "logs"

    @property
    def state_dir(self) -> Path: return self.base_dir / "State"

    @property
    def ai_cache_dir(self) -> Path: return self.state_dir / "ai_cache"

    @property
    def whisper_models_dir(self) -> Path: return self.models_dir / "whisper_models"

    @property
    def mediapipe_dir(self) -> Path: return self.models_dir / "mediapipe_models"

    @property
    def log_file(self) -> Path: return self.logs_dir / "app.log"

    @property
    def cookie_file(self) -> Path: return self.temp_dir / "cookies.txt"

    @property
    def prompt_file(self) -> Path: return self.base_dir / "resources" / "prompts" / "gemini_prompt.txt"

    @property
    def face_landmarker_file(self) -> Path: return self.mediapipe_dir / "face_landmarker.task"

    @property
    def env_file(self) -> Path: return self.base_dir / ".env"

    @property
    def raw_ai_filename(self) -> str: return "raw_ai_response.json"

    @property
    def summary_filename(self) -> str: return "summary.json"

    @property
    def state_filename(self) -> str: return "project_state.json"

    @property
    def all_directories(self) -> List[Path]:
        """
        Menyaring semua path yang merupakan direktori.
        Digunakan oleh ManagerService untuk pembuatan folder otomatis.
        """
        return [
            self.temp_dir, self.output_dir, self.models_dir,
            self.fonts_dir, self.logs_dir, self.state_dir, self.ai_cache_dir,
            self.whisper_models_dir, self.mediapipe_dir
        ]

@dataclass
class SubtitleConfig:
    """Konfigurasi untuk subtitle dan styling (ASS)."""
    font_name: str = "Poppins-Bold"
    font_size: int = 14
    margin_v: int = 50
    primary_color: str = "&H00FFFFFF"    # Putih
    secondary_color: str = "&H0000FFFF"  # Kuning
    outline_color: str = "&H00000000"    # Hitam
    back_color: str = "&H80000000"       # Transparan
    karaoke_chunk_size: int = 1
    bold: int = 1        # 1 = True, 0 = False (ASS Format)
    italic: int = 0      # 1 = True, 0 = False

@dataclass
class WhisperConfig:
    """Konfigurasi untuk model Whisper."""
    language: str = "id"
    initial_prompt: str = "Transkrip berikut adalah video YouTube berbahasa Indonesia. Gunakan ejaan baku dan tanda baca yang benar."
    use_batched_pipeline: bool = True
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

@dataclass
class AppConfig():
    """Konfigurasi global aplikasi."""
    paths: AppPaths = field(default_factory=AppPaths)
    
    def __post_init__(self):
        load_dotenv(self.paths.env_file)

    # Model AI Preferences
    gemini_models: List[str] = field(default_factory=lambda: [
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-flash-lite-latest", 
        "gemini-2.5-pro",
        "gemini-3-pro-preview",      
        "gemini-pro-latest",
    ])
    
    # FFmpeg & Hardware Acceleration
    ffmpeg_encoder_preference: Optional[str] = None # 'h264_nvenc', 'libx264', dll.
    
    # Sub-Configs
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)