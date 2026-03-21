from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

import uuid
import re

class ClipValidator:
    """Validator untuk class Clip."""
    @staticmethod
    def validate(clip):
        """Melakukan validasi pada instance Clip."""
        if clip.start_time < 0:
            raise ValueError(f"Start time tidak boleh negatif: {clip.start_time}")
        
        if clip.end_time <= clip.start_time:
            if clip.start_time - clip.end_time < 0.1:
                clip.end_time = clip.start_time + 1.0
            else:
                raise ValueError(f"End time ({clip.end_time}) harus lebih besar dari start time ({clip.start_time})")

@dataclass
class SubtitleConfig:
    font_name: str = "Poppins Bold"
    font_size: int = 60
    primary_color: str = "&H00FFFFFF" # Putih
    outline_color: str = "&H00000000" # Hitam Transparan
    back_color: str = "&H80000000"    # Background Semi-Transparan
    bold: int = 0
    italic: int = 0
    margin_v: int = 60

@dataclass
class Clip:
    id: str
    title: str
    start_time: float
    end_time: float
    energy_score: int
    vocal_energy: str
    audio_justification: str
    description: str
    caption: str
    duration: float = field(init=False)
    # Path file fisik (opsional, diisi oleh Application/Infrastructure layer saat proses berjalan)
    raw_path: Optional[str] = None
    tracked_path: Optional[str] = None
    final_path: Optional[str] = None
    
    def __post_init__(self):
        """Validasi Invarian: Memastikan state objek selalu valid setelah inisialisasi."""
        ClipValidator.validate(self)
        self.duration = round(self.end_time - self.start_time, 2)

    @staticmethod
    def sanitize_string(name: str) -> str:
        """Membersihkan string agar aman digunakan sebagai nama file/folder."""
        import re
        raw_safe = re.sub(r'[^\w\s\-_]', '', name).strip()
        # Ganti beberapa spasi atau karakter whitespace lainnya menjadi satu spasi tunggal
        return re.sub(r'\s+', ' ', raw_safe)

    @property
    def safe_filename(self) -> str:
        """Mengembalikan nama file yang aman untuk sistem operasi (Business Rule)."""
        safe_title = re.sub(r'[^\w\s\-_]', '', self.title).strip()
        safe_title = re.sub(r'\s+', ' ', safe_title)
        return f"{self.id[:8]}_{safe_title}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Clip':
        """Factory method untuk membuat instance Clip dari dictionary."""
        return cls(
            id=data.get('id') or str(uuid.uuid4()),
            title=data.get('title', 'Untitled'),
            start_time=float(data.get('start_time', 0.0)),
            end_time=float(data.get('end_time', 0.0)),
            energy_score=int(data.get('energy_score', 0)),
            vocal_energy=data.get('vocal_energy', 'Unknown'),
            audio_justification=data.get('audio_justification', ''),
            description=data.get('description', ''),
            caption=data.get('caption', ''),
            # Path fisik bisa jadi None saat di-load
            raw_path=data.get('raw_path'),
            tracked_path=data.get('tracked_path'),
            final_path=data.get('final_path')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Mengonversi instance Clip ke dictionary untuk serialisasi."""
        data = asdict(self)
        return data

    @classmethod
    def create_manual(cls, index: int, start_time: float, end_time: float) -> 'Clip':
        """Factory method untuk membuat klip manual dengan nilai default domain."""
        return cls(
            id=f"manual_{index}",
            title=f"Manual Clip {index + 1}",
            start_time=start_time,
            end_time=end_time,
            energy_score=0,
            vocal_energy="N/A",
            audio_justification="Manual",
            description="Manual timestamp",
            caption=""
        )

@dataclass
class VideoSummary:
    video_title: str
    audio_energy_profile: str
    clips: List[Clip] = field(default_factory=list)

@dataclass
class TranscriptionWord:
    word: str
    start: float
    end: float
    probability: float

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: List[TranscriptionWord] = field(default_factory=list)

@dataclass
class TrackResult:
    tracked_video: str
    width: int
    height: int
