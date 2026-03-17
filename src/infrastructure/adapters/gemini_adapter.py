import json
import logging
import time
import re
import uuid
from pathlib import Path
from typing import Optional, List, Union
import dataclasses

from google import genai
from google.genai import types

from src.domain.interfaces import IContentAnalyzer
from src.domain.models import VideoSummary, Clip

class GeminiAdapter(IContentAnalyzer):
    def __init__(self, api_key: str, model_names: List[str]):
        self.api_key = api_key
        # Pastikan format nama model benar (diawali 'models/')
        self.model_names = [f"models/{m}" if not m.startswith("models/") else m for m in model_names]
        self.client: genai.Client = genai.Client(api_key=self.api_key)

    @staticmethod
    def check_key_validity(key: str) -> bool:
        """Memeriksa apakah API Key valid dengan request ringan."""
        try:
            client = genai.Client(api_key=key)
            next(iter(client.models.list(config={'page_size': 1})), None)
            return True
        except Exception:
            return False

    def _clean_json_text(self, text: str) -> str:
        """Membersihkan markdown code blocks dari string JSON."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text.strip(), re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def _upload_and_process_audio(self, audio_path: Path) -> types.File:
        """
        Mengunggah file audio ke server Gemini dan menunggu hingga statusnya 'ACTIVE'.

        Raises:
            TimeoutError: Jika proses indexing melebihi batas waktu.
            ValueError: Jika file gagal diproses dan statusnya bukan 'ACTIVE'.
        """
        logging.debug(f"Mengunggah file audio ke Gemini: {audio_path.name}...")
        uploaded_file = self.client.files.upload(
            file=audio_path, # SDK terbaru mendukung argumen 'path' secara langsung
            config=types.UploadFileConfig(
                display_name=audio_path.name,
                mime_type='audio/wav'
            )
        )
        
        # Tunggu proses indexing
        start_wait = time.time()
        while uploaded_file.state == "PROCESSING":

            if time.time() - start_wait > 600: 
                raise TimeoutError("Timeout: Proses indexing audio terlalu lama.")
            time.sleep(2)
            if not uploaded_file.name:
                raise ValueError("File name is missing during processing")
            uploaded_file = self.client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state != "ACTIVE":
            raise ValueError(f"Gagal memproses audio. Status: {uploaded_file.state}")
        
        logging.debug(f"✅ Audio {uploaded_file.name} berhasil diproses.")
        return uploaded_file

    def _generate_clip_schema(self) -> types.Schema:
        """
        Membuat schema Gemini secara dinamis berdasarkan dataclass Clip.
        Menghindari duplikasi definisi struktur data.
        """
        properties = {}
        required_fields = []

        type_mapping = {
            str: types.Type.STRING,
            int: types.Type.INTEGER,
            float: types.Type.NUMBER
        }

        for field in dataclasses.fields(Clip):
            # Skip field internal yang opsional (path fisik)
            if field.name in ['raw_path', 'tracked_path', 'final_path', 'id']:
                continue
            
            gemini_type = type_mapping.get(field.type, types.Type.STRING)
            properties[field.name] = types.Schema(type=gemini_type)
            required_fields.append(field.name)

        return types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=required_fields
        )

    def analyze_content(self, transcript: str, audio_path: str, prompt: str) -> VideoSummary:
        """
        Menganalisis konten menggunakan Gemini dan mengembalikan objek domain VideoSummary.
        """
        audio_file_path = Path(audio_path)
        uploaded_file: Optional[types.File] = None

        try:
            request_parts: List[types.Part] = []

            if audio_file_path.exists():
                uploaded_file = self._upload_and_process_audio(audio_file_path)
                request_parts.append(types.Part({
                    "file_data": {
                        "file_uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type
                    }
                }))

            if transcript:
                request_parts.append(types.Part.from_text(text=transcript))

            request_parts.append(types.Part.from_text(text=f"Instruction:\n{prompt}"))

            summary_schema = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "video_title": types.Schema(type=types.Type.STRING),
                    "audio_energy_profile": types.Schema(type=types.Type.STRING),
                    "clips": types.Schema(
                        type=types.Type.ARRAY,
                        items=self._generate_clip_schema()
                    )
                },
                required=["video_title", "audio_energy_profile", "clips"]
            )

            # 4. Request ke Gemini dengan Fallback Strategy
            last_exception = None
            
            for model_target in self.model_names:
                try:
                    logging.debug(f"🧠 Mengirim permintaan ke Gemini ({model_target})...")
                    
                    response: types.GenerateContentResponse = self.client.models.generate_content(
                        model=model_target,
                        contents=types.Content(
                            role="user",
                            parts=request_parts
                        ),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=summary_schema
                        )
                    )

                    clean_text = self._clean_json_text(str(response.text))
                    data = json.loads(clean_text)

                    clips_list = []
                    for c_data in data.get('clips', []):
                        clips_list.append(Clip.from_dict(c_data))

                    logging.info(f"✅ Analisis berhasil menggunakan model: {model_target}")
                    return VideoSummary(
                        video_title=data.get('video_title', 'Unknown Video'),
                        audio_energy_profile=data.get('audio_energy_profile', ''),
                        clips=clips_list
                    )

                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg:
                        logging.warning(f"⚠️ Model {model_target} sibuk/limit: {e}. Mencoba model berikutnya...")
                        last_exception = e
                        time.sleep(2)
                        continue
                    else:
                        raise e
            
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Analisis konten gagal: Tidak ada respons sukses dari daftar model yang tersedia.")

        except Exception as e:
            logging.error(f"Gemini Adapter Error: {e}")
            raise

        finally:
            if uploaded_file:
                try:
                    if not uploaded_file.name:
                        raise ValueError("File name is missing during processing")
                    self.client.files.delete(name=uploaded_file.name)
                    logging.info(f"🗑️ File {uploaded_file.name} dibersihkan dari server.")
                except Exception as e:
                    logging.warning(f"Gagal membersihkan file di server Gemini ({uploaded_file.name}): {e}")