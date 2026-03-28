from pathlib import Path
from typing import Optional, List, NoReturn, Union, Any
import time

from google import genai
from google.genai import types

from src.application.context import SessionContext
from src.domain.interfaces import IGeminiAdapter, ILogger
from src.domain.exceptions import (
    AnalysisError, 
    AuthenticationError, 
    QuotaExceededError, 
    ContentPolicyViolationError
)

class GeminiAdapter(IGeminiAdapter):
    def __init__(self, model_names: List[str], logger: ILogger):
        self.logger = logger
        
        # Mendukung list model untuk failover
        self.model_names = [f"models/{m}" if not m.startswith("models/") else m for m in model_names]
        
        self.retry_options = types.HttpRetryOptions(
            attempts=5,
            initial_delay=0.5,
            max_delay=20.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )

    # --- INTERNAL HELPERS ---

    def _ensure_client(self, ctx: SessionContext) -> genai.Client:
        """Lazy initialization dan reuse client instance untuk performa."""
        if not ctx.api_key:
            raise AuthenticationError("Gemini API Key kosong.")
        
        ctx.logger.debug("🔌 Menginisialisasi Gemini Client baru...")
        return genai.Client(
            api_key=ctx.api_key,
            http_options=types.HttpOptions(
                timeout=60000,
                retry_options=self.retry_options
            )
        )

    def _wait_for_file_active(self, ctx: SessionContext, client: genai.Client, file_name: str) -> Any:
        """Menunggu file menjadi ACTIVE dengan exponential backoff."""
        poll_interval = 1.0 
        while True:
            file_ref = client.files.get(
                name=file_name,
                config=types.GetFileConfig(http_options=types.HttpOptions(retry_options=self.retry_options))
            )
            
            if file_ref.state == types.FileState.ACTIVE:
                ctx.logger.info(f"✅ File siap (ACTIVE): {file_ref.name}")
                return file_ref
            
            if file_ref.state == types.FileState.FAILED:
                raise AnalysisError(f"File processing failed: {file_ref.error.message if file_ref.error else 'Unknown'}")
            
            ctx.logger.debug(f"⏳ Status: {file_ref.state} (Cek kembali dalam {poll_interval}s)")
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10.0)

    def _prepare_payload(self, prompt: str, file_obj: Any = None) -> List[types.Part]:
        """Modularisasi persiapan data multimodal."""
        parts = []
        if file_obj:
            parts.append(types.Part(
                file_data=types.FileData(file_uri=file_obj.uri, mime_type=file_obj.mime_type)
            ))
        parts.append(types.Part.from_text(text=prompt))
        return parts

    def _handle_gemini_error(self, ctx: SessionContext, e: Exception, context: str) -> NoReturn:
        """Menerjemahkan error SDK ke Domain Exceptions."""
        err_msg = str(e).lower()
        if isinstance(e, (AnalysisError, AuthenticationError, QuotaExceededError, ContentPolicyViolationError)):
            raise e

        ctx.logger.error(f"❌ Gemini Error ({context}): {err_msg}")
        if any(x in err_msg for x in ["401", "403", "api key not valid"]):
             raise AuthenticationError(f"Gagal Autentikasi: {err_msg}", original_exception=e)
        if any(x in err_msg for x in ["429", "exhausted", "quota"]):
             raise QuotaExceededError(f"Kuota Habis: {err_msg}", original_exception=e)
        if any(x in err_msg for x in ["safety", "blocked"]):
             raise ContentPolicyViolationError(f"Konten Ditolak: {err_msg}", original_exception=e)

        raise AnalysisError(f"Gagal Analisis ({context}): {err_msg}", original_exception=e)

    # --- PUBLIC METHODS ---

    def upload_file(self, ctx: SessionContext, file_path: str) -> Any:
        """Mengunggah file dan menunggu status ACTIVE."""
        client = self._ensure_client(ctx)
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise AnalysisError(f"File tidak ditemukan: {file_path}")

        try:
            uploaded_file = client.files.upload(
                file=path_obj,
                config=types.UploadFileConfig(
                    display_name=path_obj.name,
                    http_options=types.HttpOptions(timeout=300000, retry_options=self.retry_options)
                )
            )
            ctx.logger.info(f"⏳ File terupload: {uploaded_file.name}. Menunggu pemrosesan...")
            file_resource_name = uploaded_file.name
            if not file_resource_name:
                raise AnalysisError("Gagal mendapatkan nama file dari server Google.")
            return self._wait_for_file_active(ctx, client, file_resource_name)
        except Exception as e:
            self._handle_gemini_error(ctx, e, "upload_file")

    def generate_content(self, ctx: SessionContext, prompt: str, file_obj: Any = None, response_schema: Any = None) -> Union[str, Any]:
        """Eksekusi analisis dengan strategi Failover Model."""
        client = self._ensure_client(ctx)
        parts = self._prepare_payload(prompt, file_obj)
        
        # Konfigurasi JSON jika ada schema
        gen_config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=180000)
        )
        if response_schema:
            gen_config.response_mime_type = "application/json"
            gen_config.response_schema = response_schema

        last_error = None
        # Mencoba setiap model dalam list satu per satu
        for model_name in (self.model_names or ["models/gemini-1.5-flash"]):
            try:
                ctx.logger.debug(f"🧠 Menganalisis dengan {model_name}...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(role="user", parts=parts),
                    config=gen_config
                )

                if response_schema and response.parsed:
                    return response.parsed
                if not response.text:
                    raise AnalysisError("Respon kosong dari model.")
                return response.text

            except Exception as e:
                last_error = e
                if "safety" in str(e).lower() or "blocked" in str(e).lower():
                    break
                ctx.logger.warning(f"⚠️ {model_name} bermasalah. Mencoba model cadangan...")
                continue

        if last_error:
            self._handle_gemini_error(ctx, last_error, "generate_content_all_models_failed")
        else:
            error_fallback = AnalysisError("Semua model gagal tanpa pengecekan spesifik.")
            self._handle_gemini_error(ctx, error_fallback, "generate_content_unknown_failure")

    def delete_file(self, ctx: SessionContext, file_name: str) -> None:
        """Menghapus file dari server."""
        try:
            client = self._ensure_client(ctx)
            client.files.delete(name=file_name)
            ctx.logger.debug(f"🗑️ File dihapus: {file_name}")
        except Exception as e:
            ctx.logger.warning(f"⚠️ Gagal menghapus file {file_name}: {e}")

    def close(self) -> None:
        """Pembersihan resource (Stateless)."""
        pass