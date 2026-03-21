from pathlib import Path
from typing import Optional, List, NoReturn

from google import genai
from google.genai import types

from src.domain.interfaces import IContentAnalyzer, ITextProcessor, IRetryHandler, ILogger
from src.domain.models import VideoSummary, Clip
from src.domain.exceptions import (
    AnalysisError, 
    AuthenticationError, 
    QuotaExceededError, 
    ContentPolicyViolationError
)

class GeminiAdapter(IContentAnalyzer):
    def __init__(self, api_key: str, model_names: List[str], text_processor: ITextProcessor, retry_handler: IRetryHandler, logger: ILogger):
        self.api_key = api_key
        self.text_processor = text_processor
        self.retry_handler = retry_handler
        self.logger = logger

        self.model_names = [f"models/{m}" if not m.startswith("models/") else m for m in model_names]
        self.client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def check_key_validity(key: str) -> bool:
        """Memeriksa apakah API Key valid dengan request ringan."""
        try:
            client = genai.Client(api_key=key)
            next(iter(client.models.list(config={'page_size': 1})), None)
            return True
        except Exception:
            return False

    def _upload_and_process_audio(self, audio_path: Path) -> types.File:  # UrllibDownloader:
        """
        Mengunggah file audio ke server Gemini dan menunggu hingga statusnya 'ACTIVE'.

        Raises:
            TimeoutError: Jika proses indexing melebihi batas waktu.
            ValueError: Jika file gagal diproses dan statusnya bukan 'ACTIVE'.
        """
        if not self.client:
            raise ValueError("Gemini Client belum diinisialisasi.")
        
        client = self.client
        self.logger.debug(f"Mengunggah file audio ke Gemini: {audio_path.name}...")
        uploaded_file = client.files.upload(
            file=audio_path,  # SDK terbaru mendukung argumen 'path' secara langsung
            config=types.UploadFileConfig(
                display_name=audio_path.name,
                mime_type='audio/wav'
            )
        )

        # Tunggu proses indexing
        def check_file_status():
            nonlocal uploaded_file
            if not uploaded_file.name:
                raise ValueError("File name is missing during processing")
            uploaded_file = client.files.get(name=uploaded_file.name)
            if uploaded_file.state != "ACTIVE":
                raise ValueError(f"Gagal memproses audio. Status: {uploaded_file.state}")
            return uploaded_file

        uploaded_file = self.retry_handler.execute(check_file_status)
        self.logger.debug(f"✅ Audio {uploaded_file.name} berhasil diproses.")

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

        for field_name in Clip.__dataclass_fields__:

            if field_name in ['raw_path', 'tracked_path', 'final_path', 'id']:
             continue
            
            gemini_type = type_mapping.get(Clip.__dataclass_fields__[field_name].type, types.Type.STRING)
            properties[field_name] = types.Schema(type=gemini_type)
            required_fields.append(field_name)

        return types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=required_fields
        )

    def _generate_content_api_call(self, model_target: str, request_parts: List[types.Part], schema: types.Schema) -> types.GenerateContentResponse:
        """Wrapper untuk panggilan API generate_content agar bisa di-retry."""
        if not self.client:
            raise ValueError("Gemini Client belum diinisialisasi.")
            
        return self.client.models.generate_content(
            model=model_target,
            contents=types.Content(
                role="user",
                parts=request_parts
            ),
            config=types.GenerateContentConfig(
                response_schema=schema
            )
        )

    def _handle_gemini_error(self, e: Exception, context: str) -> NoReturn:
        """Menerjemahkan error Google GenAI ke Domain Exception."""
        # Jika sudah merupakan domain exception, lempar ulang
        if isinstance(e, (AnalysisError, AuthenticationError, QuotaExceededError, ContentPolicyViolationError)):
            raise e
            
        err_msg = str(e)
        self.logger.error(f"❌ Gemini Error ({context}): {err_msg}")

        # Deteksi tipe error berdasarkan pesan (SDK agnostic)
        if "401" in err_msg or "403" in err_msg or "API key not valid" in err_msg:
             raise AuthenticationError(f"Gagal Autentikasi: {err_msg}", original_exception=e)
        
        if "429" in err_msg or "Resource has been exhausted" in err_msg or "quota" in err_msg.lower():
             raise QuotaExceededError(f"Kuota Habis/Rate Limit: {err_msg}", original_exception=e)

        if "safety" in err_msg.lower() or "blocked" in err_msg.lower():
             raise ContentPolicyViolationError(f"Konten Ditolak (Safety): {err_msg}", original_exception=e)

        raise AnalysisError(f"Gagal Analisis ({context}): {err_msg}", original_exception=e)

    def analyze_content(self, transcript: str, audio_path: str, prompt: str, api_key: str = "") -> VideoSummary:
        """
        Menganalisis konten menggunakan Gemini dan mengembalikan objek domain VideoSummary.
        """
        # Prioritize key passed in method (runtime) over the one in init (singleton)
        active_key = api_key if api_key else self.api_key
        if not active_key:
            raise AuthenticationError("API Key tidak ditemukan. Pastikan sudah login atau setup environment.")
        
        # Re-initialize client if key is provided (Stateless behavior)
        self.client = genai.Client(api_key=active_key)

        audio_file_path = Path(audio_path)
        uploaded_file: Optional[types.File] = None # Inisialisasi awal untuk mencegah UnboundLocalError

        try:
            request_parts: List[types.Part] = []

            if audio_file_path.exists():
                try:
                    uploaded_file = self.retry_handler.execute(self._upload_and_process_audio, audio_file_path)
                    if uploaded_file:
                        request_parts.append(types.Part({
                            "file_data": {
                                "file_uri": uploaded_file.uri,
                                "mime_type": uploaded_file.mime_type
                            }
                        }))
                except Exception as e:
                    # Tangkap error upload spesifik dan bungkus
                    self._handle_gemini_error(e, "Upload Audio")

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
            def generate_content_with_model(model_target):
                self.logger.debug(f"🧠 Mengirim permintaan ke Gemini ({model_target})...")
                response = self.retry_handler.execute(self._generate_content_api_call, model_target, request_parts, summary_schema)
                data = self.text_processor.extract_json(str(response.text))
                clips_list = []
                for c_data in data.get('clips', []):
                    clips_list.append(Clip.from_dict(c_data))

                self.logger.info(f"✅ Analisis berhasil menggunakan model: {model_target}")
                return VideoSummary(
                    video_title=data.get('video_title', 'Unknown Video'),
                    audio_energy_profile=data.get('audio_energy_profile', ''),
                    clips=clips_list
                )

            # Perbaikan Logika Fallback Loop
            for model_target in self.model_names:
                try:                
                    return self.retry_handler.execute(generate_content_with_model, model_target)           
                except Exception as e:
                    last_exception = e
                    self.logger.warning(f"⚠️ Model {model_target} gagal/overload. Mencoba model berikutnya... Error: {e}")
                    # Continue agar model berikutnya dicoba, jangan raise di sini!
                    continue

            if last_exception:          
                # Jika semua model gagal, bungkus error terakhir
                self._handle_gemini_error(last_exception, "All Models Failed")
            
            raise AnalysisError("Analisis konten gagal: Tidak ada respons sukses dari daftar model yang tersedia.")

        except Exception as e:
            # Tangkap semua exception level atas dan pastikan terbungkus
            self._handle_gemini_error(e, "analyze_content")

        finally:
            # Cek existence uploaded_file dan name-nya sebelum akses
            if uploaded_file and hasattr(uploaded_file, 'name') and uploaded_file.name:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    self.logger.info(f"🗑️ File {uploaded_file.name} dibersihkan dari server.")
                except Exception as e:
                    self.logger.warning(f"Gagal membersihkan file di server Gemini ({uploaded_file.name}): {e}")