from pathlib import Path
from typing import Optional, List, Dict, Any

from src.application.context import SessionContext
from src.domain.interfaces import IYoutubeAdapter, IFfmpegAdapter, IUtilsCacheManager, ILogger, IProviderService, IGeminiAdapter, IWhisperAdapter
from src.domain.models import VideoSummary, Clip, ProjectState, TranscriptionSegment
from src.domain.exceptions import MediaDownloadError, RateLimitError, AnalysisError
from src.application.common.dtos import AIVideoSummarySchema

class ProviderService(IProviderService):
    def __init__(self, downloader: IYoutubeAdapter, processor: IFfmpegAdapter, analyzer: IGeminiAdapter, transcriber: IWhisperAdapter, cache_manager: IUtilsCacheManager, prompt_path: Path, logger: ILogger, ai_cache_dir: Path, raw_ai_filename: str, summary_filename: str, state_filename: str):
        self.downloader = downloader
        self.processor = processor
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.cache_manager = cache_manager
        self.prompt_path = prompt_path
        self.logger = logger
        self.ai_cache_dir = ai_cache_dir
        self.raw_ai_filename = raw_ai_filename
        self.summary_filename = summary_filename
        self.state_filename = state_filename
        self.state_manager = ProjectStateManager(cache_manager)

        self._cached_prompt : Optional[str] = None
        self._padding_seconds = 1.5

    def get_safe_folder_name(self, ctx: SessionContext, url: str) -> Optional[str]:
        return self.downloader.get_safe_title(ctx, url)

    def warmup_ai(self, ctx: SessionContext) -> None:
        """Memicu pemuatan model AI di latar belakang (Hybrid Lazy Loading)."""
        ctx.logger.debug("🧠 Menyiapkan resources AI (Whisper)...")
        self.transcriber.ensure_model(ctx)

    def close_ai(self, ctx: SessionContext) -> None:
        """Membersihkan resource AI yang digunakan oleh provider."""
        self.transcriber.close(ctx)
        self.analyzer.close()

    def get_audio_for_analysis(self, ctx: SessionContext, url: str, temp_dir: Path, filename_prefix: str) -> Path:
        """
        Memastikan file audio WAV yang siap untuk dianalisis tersedia.
        Mengatur alur: Cek Cache -> Unduh (langsung ke WAV) -> Selesai.

        Raises:
            ConnectionError: Jika download gagal.
            IOError: Jika file tidak ditemukan setelah download.
        """
        out_path = temp_dir / f"{filename_prefix}.wav"

        if out_path.exists() and out_path.stat().st_size > 10240:
            # Static logging via self.logger is okay for file-level info
            return out_path

        downloaded_audio_path_str = self.downloader.download_audio(ctx, url, str(temp_dir), filename_prefix)
        downloaded_path = Path(downloaded_audio_path_str)

        if downloaded_path.exists() and downloaded_path.suffix.lower() == '.wav':
            if downloaded_path.resolve() != out_path.resolve():                
                downloaded_path.rename(out_path)
            return out_path
        
        raise IOError("Gagal mendapatkan file audio WAV. File tidak ditemukan atau format salah setelah proses unduh.")
    

    def get_prompt_for_analysis(self, ctx: SessionContext) -> str:
        """
        Implementasi Lazy Loading murni.
        Membaca disk hanya SATU KALI selama aplikasi berjalan.
        """
        if self._cached_prompt is None:
            prompt_file = Path(self.prompt_path)
            if not prompt_file.exists():
                error_msg = f"Critical Resource Missing: Prompt file tidak ditemukan di {prompt_file}"
                ctx.logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            try:
                ctx.logger.debug(f"📖 Memuat prompt template dari disk: {prompt_file.name}")
                self._cached_prompt = prompt_file.read_text(encoding='utf-8')
            except Exception as e:
                ctx.logger.error(f"❌ Gagal membaca file prompt: {str(e)}")
                raise
        
        return self._cached_prompt
    
    def analyze_video(
        self, 
        ctx: SessionContext,
        url: str, 
        temp_dir: Path, 
        filename_prefix: str,
        cache_path: str, 
        audio_path: Optional[str] = None
    ) -> VideoSummary:
        # Gunakan cache_path (safe_name) untuk folder cache persisten
        project_cache_dir = self.ai_cache_dir / cache_path
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        
        raw_ai_cache_file = str(project_cache_dir / self.raw_ai_filename)
        summary_cache_file = str(project_cache_dir / self.summary_filename)

        # 1. Cek Tahap Akhir (Summary Ter-refine)
        cached_summary = self._load_from_cache(ctx, summary_cache_file)
        if cached_summary:
            ctx.logger.info("♻️  Menggunakan summary ter-refine dari cache.")
            return cached_summary

        # 2. STAGE FETCH: Dapatkan Data Mentah dari Gemini
        raw_data = self.cache_manager.load(ctx, raw_ai_cache_file)
        if not raw_data:
            ctx.logger.info("🧠 Memanggil Gemini API untuk analisis baru...")
            prompt = self.get_prompt_for_analysis(ctx)
            
            if audio_path is None:
                audio_path = str(self.get_audio_for_analysis(ctx, url, temp_dir, filename_prefix))

            try:
                raw_response = self._fetch_gemini_analysis_raw(ctx, prompt, audio_path)
                raw_data = raw_response.model_dump() # Simpan dalam bentuk dict/JSON
                self.cache_manager.save(ctx, raw_data, raw_ai_cache_file)
                ctx.logger.debug(f"💾 Raw AI Response disimpan ke {self.raw_ai_filename}")
            finally:
                self.analyzer.close()
        else:
            ctx.logger.info(f"♻️  Menggunakan Raw AI Response dari cache: {self.raw_ai_filename}")

        # 3. STAGE REFINE: Whisper Snapping & Refinement
        # Convert Raw Data ke Domain Model dulu untuk diproses
        summary = self._map_dto_to_domain(ctx, AIVideoSummarySchema(**raw_data))
        
        if audio_path is None:
            audio_path = str(self.get_audio_for_analysis(ctx, url, temp_dir, filename_prefix))
            
        refined_summary = self._refine_analysis_with_whisper(ctx, summary, audio_path)
        
        # Simpan hasil akhir yang sudah rapi
        self._save_to_cache(ctx, refined_summary, summary_cache_file)
        return refined_summary

    def _refine_analysis_with_whisper(self, ctx: SessionContext, summary: VideoSummary, audio_path: str) -> VideoSummary:
        """Tahap pengolahan transkripsi lokal untuk merapikan timestamp Gemini."""
        if not summary.clips:
            return summary
            
        max_duration = self.processor.get_video_duration(ctx, audio_path) or 0.0
        batch_ts = self._get_batch_clip_timestamps(summary.clips, max_duration)
        
        ctx.logger.info(f"⚡ Memulai Batch Transcription (Whisper) untuk {len(summary.clips)} segmen...")
        
        all_segments = list(self.transcriber.transcribe(
            ctx=ctx,
            audio_path=audio_path,
            clip_timestamps=batch_ts
        ))
        
        self._map_segments_to_clips(summary.clips, all_segments)
        return summary

    def _fetch_gemini_analysis_raw(self, ctx: SessionContext, prompt: str, audio_path: str) -> AIVideoSummarySchema:
        """Hanya bertugas memanggil API dan mengembalikan DTO mentah."""
        uploaded_file = self.analyzer.upload_file(ctx, file_path=str(audio_path))
        try:
            return self.analyzer.generate_content(
                ctx=ctx,
                prompt=prompt,
                file_obj=uploaded_file,
                response_schema=AIVideoSummarySchema
            )
        finally:
            if uploaded_file and hasattr(uploaded_file, 'name'):
                self.analyzer.delete_file(ctx, uploaded_file.name)

    def _get_batch_clip_timestamps(self, clips: List[Clip], max_duration: float = 0.0) -> List[float]:
        """Mengumpulkan start,end kasar dari semua klip menjadi flat list untuk Whisper."""
        ts_list = []
        for clip in clips:
            s_buffered = max(0.0, float(clip.start_time) - self._padding_seconds)
            e_buffered = float(clip.end_time) + self._padding_seconds
            
            # Clamping: Pastikan tidak melebihi durasi video
            if max_duration > 0:
                e_buffered = min(e_buffered, max_duration)
                
            ts_list.extend([s_buffered, e_buffered])
        return ts_list

    def _map_segments_to_clips(self, clips: List[Clip], all_segments: List[TranscriptionSegment]):
        """Opsi A: Mencocokkan segmen transkripsi ke objek klip berdasarkan koordinat waktu."""
        for clip in clips:
            # Range pencarian sedikit lebih luas dari buffer untuk toleransi floating point
            search_start = max(0.0, clip.start_time - self._padding_seconds - 0.5)
            search_end = clip.end_time + self._padding_seconds + 0.5
            
            # Filter segmen yang jatuh dalam range klip ini
            matching_segments = [
                seg for seg in all_segments 
                if seg.start >= search_start and seg.end <= search_end
            ]
            
            if matching_segments:
                self._snap_single_clip_to_transcript(clip, matching_segments)

    def _snap_single_clip_to_transcript(self, clip: Clip, segments: List[TranscriptionSegment]):
        """Menyelaraskan satu klip menggunakan segmen transkripsi lokal."""
        all_words = [w for seg in segments for w in seg.words]
        if not all_words:
            return

        # 1. Start Time Snapping: Cari kata pertama yang diucapkan
        # Karena Whisper sudah dipotong (clip_timestamps), kata pertama biasanya adalah awal klip yang benar
        clip.start_time = all_words[0].start
        
        # 2. End Time Snapping: Cari batas kalimat atau jeda
        # Kita cari kata yang paling mendekati estimasi end_time dari Gemini
        closest_end_idx = min(range(len(all_words)), key=lambda i: abs(all_words[i].end - clip.end_time))
        
        # Lookahead: Cari titik (.) atau jeda hening (silence) agar tidak memotong kalimat
        # Kita mulai pencarian sedikit sebelum closest_end untuk memastikan konteks
        search_start_idx = max(0, closest_end_idx - 5)
        search_limit = min(len(all_words) - 1, closest_end_idx + 20)
        final_end_time = all_words[closest_end_idx].end

        for i in range(search_start_idx, search_limit + 1):
            w = all_words[i]
            final_end_time = w.end
            # Berhenti jika menemukan tanda baca akhir kalimat
            if any(p in w.word for p in ['.', '!', '?', '"']):
                break
            # Deteksi Jeda Hening > 0.4 detik sebagai pemotong alami
            if i < len(all_words) - 1 and (all_words[i+1].start - w.end > 0.4):
                break

        clip.end_time = final_end_time + 0.15 # Sedikit tail buffer agar tidak terpotong tajam
        clip.words = all_words # Simpan hasil transkripsi untuk subtitle nanti

    def load_project_state(self, ctx: SessionContext, work_dir: str) -> ProjectState:
        """Memuat state proyek dari folder cache persisten."""
        project_cache_dir = self.ai_cache_dir / work_dir
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        state_file = project_cache_dir / self.state_filename
        return self.state_manager.load_state(ctx, state_file)

    def save_project_state(self, ctx: SessionContext, work_dir: str, state: ProjectState) -> None:
        """Menyimpan state proyek ke folder cache persisten."""
        project_cache_dir = self.ai_cache_dir / work_dir
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        state_file = project_cache_dir / self.state_filename
        self.state_manager.save_state(ctx, state, state_file)

    def _load_from_cache(self, ctx: SessionContext, path: str) -> Optional[VideoSummary]:
        """Helper internal untuk memuat JSON cache ke Domain Model."""
        data = self.cache_manager.load(ctx, path)
        if not data:
            return None
        
        try:
            clips = []
            clips = [Clip.from_dict(c_data) for c_data in data.get('clips', [])]

            return VideoSummary(
                context_keywords=data.get('context_keywords', ''),
                clips=clips
            )
        except Exception as e:
            ctx.logger.warning(f"⚠️ Struktur cache tidak valid: {e}")
            return None

    def _save_to_cache(self, ctx: SessionContext, summary: VideoSummary, path: str):
        data = {
            "context_keywords": summary.context_keywords,
            "clips": [c.to_dict() for c in summary.clips]
        }
        try:
            self.cache_manager.save(ctx, data, path)
        except Exception as e:
            ctx.logger.warning(f"⚠️ Gagal menyimpan cache analisis: {e}")

    def _map_dto_to_domain(self, ctx: SessionContext, data: AIVideoSummarySchema) -> VideoSummary:
        """Mapping objek schema Pydantic ke objek Domain VideoSummary."""
        # data adalah instance AIVideoSummarySchema
        
        if not data.clips:
             raise ValueError("No valid clips found in response")
        
        clips_list = []
        # Konversi setiap item di list clips
        for c_schema in data.clips:
            try:
                # Konversi schema object kembali ke dict agar kompatibel dengan Domain Factory
                c_data = c_schema.model_dump()
                
                # Clip.from_dict menangani default value
                clips_list.append(Clip.from_dict(c_data))
            except Exception as e:
                ctx.logger.warning(f"Skipping malformed clip data: {e}")

        if not clips_list:
            raise ValueError("No valid clips found in response")

        return VideoSummary(
            context_keywords=data.context_keywords,
            clips=clips_list
        )

class ProjectStateManager:
    """
    Helper class untuk mengelola file state proyek (project_state.json).
    Memisahkan penyimpanan status pengerjaan (Mutable) dari hasil analisis AI (Immutable).
    """
    def __init__(self, cache_manager: IUtilsCacheManager):
        self.cache_manager = cache_manager

    def load_state(self, ctx: SessionContext, path: Path) -> ProjectState:
        """Memuat state proyek dari disk. Mengembalikan state kosong jika file tidak ada."""
        data = self.cache_manager.load(ctx, str(path))
        if not data:
            return ProjectState(video_source_url="")
        
        try:
            return ProjectState.from_dict(data)
        except Exception as e:
            ctx.logger.warning(f"⚠️ Project State rusak atau usang: {e}. Membuat state baru.")
            return ProjectState(video_source_url="")

    def save_state(self, ctx: SessionContext, state: ProjectState, path: Path) -> None:
        """Menyimpan state proyek ke disk."""
        try:
            self.cache_manager.save(ctx, state.to_dict(), str(path))
        except Exception as e:
            ctx.logger.error(f"❌ Gagal menyimpan project state ke {path.name}: {e}")