from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.models import SubtitleConfig

@dataclass
class AppPaths:
    # Gunakan default_factory agar aman dan dinamis
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    
    # Folder Struktur (init=False artinya field ini diisi otomatis oleh __post_init__)
    TEMP_DIR: Path = field(init=False)
    OUTPUT_DIR: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    FILES_DIR: Path = field(init=False)
    FONTS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    LOG_FILE: Path = field(init=False)
    
    # Sub-folder Models
    WHISPER_MODELS_DIR: Path = field(init=False)
    MEDIAPIPE_DIR: Path = field(init=False)
    
    # Files
    ENV_FILE: Path = field(init=False)
    COOKIE_FILE: Path = field(init=False)
    PROMPT_FILE: Path = field(init=False)
    FACE_LANDMARKER_FILE: Path = field(init=False)
    FFMPEG_CACHE_FILE: Path = field(init=False)

    def __post_init__(self):
        self.TEMP_DIR = self.BASE_DIR / "Temp"
        self.OUTPUT_DIR = self.BASE_DIR / "Output"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.FILES_DIR = self.BASE_DIR / "files"
        self.FONTS_DIR = self.BASE_DIR / "fonts"
        
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.LOG_FILE = self.LOGS_DIR / "app.log"
        
        self.WHISPER_MODELS_DIR = self.MODELS_DIR / "whispermodels"
        self.MEDIAPIPE_DIR = self.MODELS_DIR / "mpmodels"
        
        self.ENV_FILE = self.FILES_DIR / ".env"
        self.COOKIE_FILE = self.FILES_DIR / "cookies.txt"
        self.PROMPT_FILE = self.BASE_DIR / "resources" / "prompts" / "gemini_prompt.txt"
        self.FACE_LANDMARKER_FILE = self.MEDIAPIPE_DIR / "face_landmarker.task"
        self.FFMPEG_CACHE_FILE = self.FILES_DIR / "ffmpeg_cache.json"

@dataclass
class AppConfig:
    # Pastikan menggunakan default_factory (ini yang memperbaiki error mutable default)
    paths: AppPaths = field(default_factory=AppPaths)
    
    gemini_models: List[str] = field(default_factory=lambda: [
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview"
        ]    
    )
    
    # FFmpeg Preference
    ffmpeg_encoder_preference: Optional[str] = None
    
    # Motion Tracking
    motion_window_size: int = 5
    motion_process_every_n_frames: int = 3
    
    # Captioning
    karaoke_chunk_size: int = 1
    
    # Subtitle Styling
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    
    # Whisper Model Strategy (Simple)
    whisper_model_size: str = "small" 
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"