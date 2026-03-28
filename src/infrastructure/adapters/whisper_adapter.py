from typing import List, Optional, Dict, Any, Iterable
import threading
from dataclasses import asdict

import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from src.application.context import SessionContext

from src.domain.interfaces import IWhisperAdapter, TranscriptionSegment, TranscriptionWord, ILogger, IWhisperConfig
from src.domain.exceptions import TranscriptionError

class WhisperAdapter(IWhisperAdapter):
    """
    Implementasi ITranscriber menggunakan faster-whisper.
    Bertanggung jawab murni untuk transkripsi audio ke data terstruktur.
    """

    def __init__(self, config: IWhisperConfig, model_size: str, device: str, compute_type: str, logger: ILogger, download_root: Optional[str] = None):
        self.config = config
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.logger = logger
        self.download_root = download_root
        
        self._lock = threading.Lock()
        # Model dan Pipeline tidak dimuat di awal (Lazy Loading)
        self._model: Optional[WhisperModel] = None
        self._pipeline: Optional[BatchedInferencePipeline] = None

    def ensure_model(self, ctx: SessionContext):
        """Memuat model ke memori hanya saat benar-benar dibutuhkan."""
        with self._lock:
            if self._model is not None:
                return

            ctx.logger.info(f"🧠 Loading WhisperModel ({self.model_size}) to {self.device}...")
            try:
                self._model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type, 
                    download_root=self.download_root
                )
                if self.config.use_batched_pipeline:
                    self._pipeline = BatchedInferencePipeline(model=self._model)
                ctx.logger.info("✅ WhisperModel loaded successfully.")
            except Exception as e:
                ctx.logger.error(f"Failed to initialize WhisperModel: {e}", exc_info=True)
                raise TranscriptionError(f"Gagal menginisialisasi model Whisper: {e}", original_exception=e)

    def close(self, ctx: SessionContext):
        """Membersihkan resource model dari memori."""
        with self._lock:
            if self._model is not None:
                ctx.logger.debug("🧹 Melepaskan resource WhisperModel...")
                # faster-whisper tidak memiliki method close() eksplisit, 
                # namun menghapus referensi dan mengosongkan cache CUDA sangat membantu.
                self._model = None
                self._pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @staticmethod
    def detect_hardware(logger: Optional[ILogger] = None) -> Dict[str, str]:
        """
        Mendeteksi hardware dan mengembalikan konfigurasi optimal untuk Whisper.
        Returns:
            Dict[str, str]: Berisi 'model_size', 'device', dan 'compute_type'.
        """
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
                gpu_name = props.name
                if logger: logger.info(f"🚀 AI Hardware: NVIDIA GPU ({gpu_name}) | VRAM: {vram_gb:.2f}GB")

                if vram_gb >= 10: model_size = "large-v3"
                elif vram_gb >= 4: model_size = "medium"
                else:
                    if logger: logger.warning(f"   -> VRAM < 4GB. Fallback ke model 'small' di CPU.")
                    return {'model_size': 'small', 'device': 'cpu', 'compute_type': 'int8'}
                
                if logger: logger.info(f"   -> Tier: GPU. Memilih model Whisper: {model_size}")
                return {'model_size': model_size, 'device': 'cuda', 'compute_type': 'float16'}
            else:
                if logger: logger.warning("⚠️ GPU tidak terdeteksi untuk AI. Menggunakan CPU.")
                return {'model_size': 'small', 'device': 'cpu', 'compute_type': 'int8'}
        except Exception as e:
            if logger: logger.warning(f"⚠️ Gagal mendeteksi GPU (Torch error: {e}). Default ke CPU.")
            return {'model_size': 'small', 'device': 'cpu', 'compute_type': 'int8'}

    def _segment_to_dict(self, segment: Any) -> TranscriptionSegment:
        """Mengubah objek Segment dari faster-whisper menjadi dictionary."""
        words_list: List[TranscriptionWord] = []
        if segment.words:
            for w in segment.words:
                words_list.append(TranscriptionWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability
                ))
        return TranscriptionSegment(
            start=segment.start, end=segment.end, text=segment.text, words=words_list
        )

    def transcribe(
        self, 
        ctx: SessionContext,
        audio_path: str, 
        initial_prompt: Optional[str] = None,
        clip_timestamps: Optional[List[float]] = None
    ) -> Iterable[TranscriptionSegment]:
        """
        Mentranskripsi file audio dan mengembalikan hasilnya sebagai Generator (Lazy Evaluation).
        """
        # Trigger pemuatan model jika belum ada
        self.ensure_model(ctx)

        use_batched = self.config.use_batched_pipeline

        # Jika ada targeted clip_timestamps, kita paksa non-batched untuk presisi word-level
        if clip_timestamps:
            ctx.logger.debug(f"🎯 Targeted Batch Transcription: {len(clip_timestamps)//2} segmen terdeteksi.")
            use_batched = False

        ctx.logger.debug(f"🎙️ Memulai transkripsi: {audio_path} | Mode: {'Batched' if use_batched else 'Sequential'}")

        try:
            if use_batched and self._pipeline:
                # Batched Mode (Default untuk full audio)
                segments, info = self._pipeline.transcribe(audio_path, batch_size=3, word_timestamps=True) 
            elif self._model:
                # Sequential Mode (Targeted & Detailed)
                # Menggunakan argumen eksplisit, bukan dictionary unpacking
                # clip_timestamps mengharapkan str atau list, default library adalah "0"
                active_clip_ts = clip_timestamps if clip_timestamps is not None else "0"
                segments, info = self._model.transcribe(
                    audio=audio_path,
                    word_timestamps=True,
                    initial_prompt=initial_prompt or self.config.initial_prompt,
                    language=self.config.language,
                    clip_timestamps=active_clip_ts,
                    log_prob_threshold=self.config.log_prob_threshold,
                    compression_ratio_threshold=self.config.compression_ratio_threshold
                )
            else:
                # Pylance Guard: Memastikan variabel segments & info tidak unbound
                raise TranscriptionError("Model Whisper tidak diinisialisasi dengan benar.")
            
            ctx.logger.info(f"📝 Transkripsi berjalan. Bahasa terdeteksi: {info.language} (Prob: {info.language_probability:.2f})")
            
            # Menggunakan Generator untuk efisiensi memori (Yield per segmen)
            for seg in segments:
                yield self._segment_to_dict(seg)

        except Exception as e:
            ctx.logger.error(f"Error during transcription: {e}", exc_info=True)
            raise TranscriptionError(f"Gagal melakukan transkripsi audio: {e}", original_exception=e)