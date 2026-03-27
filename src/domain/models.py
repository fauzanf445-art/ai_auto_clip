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
class TranscriptionWord:
    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionWord':
        return cls(**data)

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
    words: List[TranscriptionWord] = field(default_factory=list)
    
    def __post_init__(self):
        """Validasi Invarian: Memastikan state objek selalu valid setelah inisialisasi."""
        ClipValidator.validate(self)

    @property
    def duration(self) -> float:
        """Menghitung durasi secara dinamis (Computed Property)."""
        return round(self.end_time - self.start_time, 2)

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
    context_keywords: str
    clips: List[Clip] = field(default_factory=list)

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: List[TranscriptionWord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSegment':
        words_data = data.get("words", [])
        return cls(
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            text=data.get("text", ""),
            words=[TranscriptionWord.from_dict(w) for w in words_data]
        )

@dataclass
class TrackResult:
    tracked_video: str
    width: int
    height: int

@dataclass
class ClipState:
    """Menyimpan status pengerjaan dan lokasi file fisik dari sebuah Clip."""
    id: str
    raw_path: Optional[str] = None
    tracked_path: Optional[str] = None
    final_path: Optional[str] = None
    status: str = "PENDING"

@dataclass
class ProjectState:
    """Manajemen state global untuk tracking progress pemrosesan video."""
    video_source_url: str
    clip_states: Dict[str, ClipState] = field(default_factory=dict)

    def get_clip_state(self, clip_id: str) -> ClipState:
        if clip_id not in self.clip_states:
            self.clip_states[clip_id] = ClipState(id=clip_id)
        return self.clip_states[clip_id]
        
    def update_state(self, clip_id: str, **kwargs):
        state = self.get_clip_state(clip_id)
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_source_url": self.video_source_url,
            "clip_states": {cid: asdict(s) for cid, s in self.clip_states.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectState':
        instance = cls(video_source_url=data.get("video_source_url", ""))
        states = data.get("clip_states", {})
        for cid, s_data in states.items():
            instance.clip_states[cid] = ClipState(**s_data)
        return instance
