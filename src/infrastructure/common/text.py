import json
import re
from typing import Dict, Any

from src.domain.interfaces import ITextProcessor

class RegexTextProcessor(ITextProcessor):
    """Implementasi text processor menggunakan Regex."""
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Membersihkan markdown code blocks dari string untuk mengekstrak JSON murni."""
        # Pola regex untuk menangkap konten di dalam ```json ... ``` atau ``` ... ```
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text.strip(), re.DOTALL)
        
        clean_text = match.group(1) if match else text.strip()
        return json.loads(clean_text)