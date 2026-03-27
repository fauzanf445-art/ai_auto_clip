from typing import List
from pydantic import BaseModel, Field

# --- Definisi Schema untuk Structured Output (DTO) ---
class AIClipSchema(BaseModel):
    title: str = Field(description="Short, engaging title for the clip")
    start_time: float = Field(description="Start timestamp in seconds")
    end_time: float = Field(description="End timestamp in seconds")
    energy_score: int = Field(description="Vocal energy score 0-100")
    vocal_energy: str = Field(description="Description of energy (High, Excited, etc)")
    audio_justification: str = Field(description="Reason for selection")
    caption: str = Field(description="Viral-worthy caption in Indonesian")

class AIVideoSummarySchema(BaseModel):
    context_keywords: str = Field(description="Summary of topics and style")
    clips: List[AIClipSchema]
