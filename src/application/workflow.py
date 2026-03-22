from pathlib import Path
from typing import List, Tuple, Optional

# Import Config & UI
from src.config import AppConfig
from src.domain.models import Clip
from src.application.context import SessionContext
from src.domain.interfaces import TrackResult, ILogger, IProviderService, IEditorService, IWorkspaceFactory
from src.domain.exceptions import HSUAIClipError, MediaDownloadError, AnalysisError

class Workflow:
    def __init__(
        self,
        config: AppConfig,
        provider: IProviderService,
        editor: IEditorService,
        manager_factory: IWorkspaceFactory,
        logger: ILogger
    ):
        self.config = config
        self.provider = provider
        self.editor = editor
        self.manager_factory = manager_factory
        self.logger = logger
        self.prompt_template = self._load_prompt_template()

    def _process_url(self, ctx: SessionContext, url: str, work_dir: Path, safe_name: str):
        """Orkestrasi utama: Memproses URL video dari awal hingga akhir."""
        
        # 1. Identifikasi Klip (Manual atau AI)
        clips = self._identify_clips(ctx, url, work_dir)
        if not clips:
            ctx.ui.show_error("Tidak ada klip yang ditemukan atau dibuat.")
            return

        # 2. Ekstraksi (Download & Potong)
        raw_clip_paths = self._cut_raw_clips(ctx, clips, url, work_dir)
        if not raw_clip_paths:
            raise MediaDownloadError("Gagal memotong klip mentah sama sekali. Periksa koneksi atau log.")
            
        # 3. Pemrosesan (Tracking & Rendering)
        tracked_results = self._track_clips(ctx, raw_clip_paths, work_dir)
        final_clips = self._render_final_clips(ctx, tracked_results, work_dir, safe_name)

        # 4. Finalisasi (UI & Cleanup)
        self._finalize_processing(ctx, safe_name, final_clips)

    def run(self, ctx: SessionContext, url: str):
        """Menjalankan pipeline lengkap."""
        try:
            ctx.ui.show_info("🚀 [STEP] Inisialisasi...")
            video_metadata = self.provider.get_safe_folder_name(url)
            if not video_metadata:
                raise MediaDownloadError("Gagal mengambil metadata video (Judul/ID). Pastikan URL valid atau cookies terkonfigurasi.")

            with self.manager_factory.create(video_metadata) as (safe_name, work_dir):                
                ctx.ui.show_info(f"   -> Working Directory: {work_dir}")
                self._process_url(ctx, url, work_dir, safe_name)

        except HSUAIClipError as e:
            self.logger.error(f"Pipeline Error: {e}")
            ctx.ui.show_error(str(e))
        except Exception as e:
            self.logger.error("Orchestrator Error", exc_info=True)
            ctx.ui.show_error(str(e))

    def _load_prompt_template(self) -> str:
        """Memuat prompt template dari file konfigurasi dengan prinsip Fail Fast."""
        prompt_path = self.config.paths.PROMPT_FILE
        
        if not prompt_path.exists():
            error_msg = f"Critical Resource Missing: Prompt file tidak ditemukan di {prompt_path}"
            self.logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
            
        return prompt_path.read_text(encoding='utf-8')

    def _identify_clips(self, ctx: SessionContext, url: str, work_dir: Path) -> List[Clip]:
        """Menentukan sumber klip: Input manual user atau Analisis AI."""
        ctx.ui.show_info("🚀 [STEP] Analisis Konten...")
        
        # Coba mode manual terlebih dahulu
        manual_clips = self._try_get_manual_clips()
        if manual_clips:
            return manual_clips
            
        # Fallback ke mode AI
        return self._perform_ai_analysis(ctx, url, work_dir)

    def _try_get_manual_clips(self) -> Optional[List[Clip]]:
        """Mencoba mendapatkan input timestamp dari user."""
        # Logika interaktif ini sebaiknya di Controller, tapi untuk transisi:
        # Kita lewati manual input jika UI tidak interaktif atau implementasi sederhana
        # self.ui.get_input("Masukkan timestamp manual") -> Logic parsing pindah sini
        # Untuk refactoring tahap ini, kita matikan dulu fitur manual via Workflow
        # agar fokus pada pembersihan UI.
        return None

    def _perform_ai_analysis(self, ctx: SessionContext, url: str, work_dir: Path) -> List[Clip]:
        """Menjalankan analisis AI lengkap (Transkrip + Audio + Gemini)."""
        ctx.ui.show_info("   -> Mode AI: Menganalisis video untuk klip potensial...")
        
        # Membuat subfolder 'source' untuk menyimpan file mentah dan cache analisis
        source_dir = work_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = self.provider.get_transcript(url, temp_dir=str(source_dir), filename_prefix="transcript")
        
        # Baca konten transkrip dari Path
        transcript_text = ""
        if transcript_path.exists() and transcript_path.stat().st_size > 0:
            # Gunakan errors='ignore' untuk menghindari crash karena encoding karakter aneh
            transcript_text = transcript_path.read_text(encoding='utf-8', errors='ignore')

        audio_wav_path = self.provider.prepare_media_for_analysis(url, source_dir, "full_audio")
        
        if not audio_wav_path:
            raise MediaDownloadError("Gagal menyiapkan audio untuk analisis.")

        prompt = self.prompt_template
        cache_path = str(source_dir / "summary.json")
        
        summary = self.provider.analyze_video(
            transcript=transcript_text,
            audio_path=str(audio_wav_path),
            prompt=prompt,
            cache_path=cache_path,
            api_key=ctx.api_key  # Passing API Key dari Context
        )
        ctx.ui.show_info(f"   -> AI menemukan {len(summary.clips)} klip potensial.")
        return summary.clips

    def _cut_raw_clips(self, ctx: SessionContext, clips: List[Clip], url: str, work_dir: Path) -> List[Path]:
        """
        Memotong klip mentah dari stream video. 
        Menggunakan strategi 'Direct Segment Download' dengan pemaksaan CFR.
        """
        ctx.ui.show_info("🚀 [STEP] Downloading & CFR Conversion...")
        # Note: Kita menggunakan URL asli YouTube dan yt-dlp downloader

        raw_clips_dir = work_dir / "raw_clips"
        created_clip_paths = self.editor.batch_create_clips(
            clips=clips,
            source_url=url,
            output_dir=raw_clips_dir,
            progress_reporter=ctx.progress_reporter
        )
        ctx.ui.show_info(f"   -> {len(created_clip_paths)} dari {len(clips)} klip berhasil dipotong.")
        return created_clip_paths

    def _track_clips(self, ctx: SessionContext, raw_clip_paths: List[Path], work_dir: Path) -> List[Tuple[Path, TrackResult]]:
        """Menjalankan motion tracking pada setiap klip mentah."""
        ctx.ui.show_info("🚀 [STEP] Motion Tracking (MediaPipe)...")
        tracked_dir = work_dir / "tracked_clips"
        tracked_results: List[Tuple[Path, TrackResult]] = []
        
        # Outer progress bar untuk setiap klip
        if ctx.progress_reporter:
            iterator = ctx.progress_reporter.sequence(raw_clip_paths, desc="Overall Tracking", unit="clip")
        else:
            iterator = raw_clip_paths

        for clip_path in iterator:
            self.logger.debug(f"   -> Tracking klip: {clip_path.name}")
            output_tracked = tracked_dir / f"{clip_path.name}"
            
            try:
                # Eksekusi Sekuensial: Mencegah OOM dengan memproses satu per satu
                result = self.editor.track_subject(str(clip_path), str(output_tracked),progress_reporter=ctx.progress_reporter)
                tracked_results.append((clip_path, result))
            except Exception as e:
                self.logger.error(f"❌ Gagal tracking klip {clip_path.name}: {e}")
                ctx.ui.show_info(f"   -> ⚠️ Skip klip {clip_path.name} karena error tracking.")

        return tracked_results

    def _render_final_clips(self, ctx: SessionContext, tracked_results: List[Tuple[Path, TrackResult]], work_dir: Path, safe_name: str) -> List[Path]:
        """Membuat subtitle dan merender video final."""
        ctx.ui.show_info("🚀 [STEP] Captioning & Rendering Final...")
        
        return self.editor.batch_render(
            tracked_results=tracked_results,
            work_dir=work_dir,
            output_dir=self.config.paths.OUTPUT_DIR / safe_name,
            progress_reporter=ctx.progress_reporter
        )

    def _finalize_processing(self, ctx: SessionContext, safe_name: str, final_clips: List[Path]):
        """Menampilkan hasil sukses dan membersihkan output lama."""
        output_folder = self.config.paths.OUTPUT_DIR / safe_name
        ctx.ui.show_info("="*40)
        ctx.ui.show_info("✨ PROSES SELESAI!")
        ctx.ui.show_info("="*40)
        ctx.ui.show_info(f"📂 Folder Output: {output_folder}")
        if final_clips:
            ctx.ui.show_info(f"🎬 {len(final_clips)} Klip Berhasil Dibuat:")
            for c in final_clips:
                ctx.ui.show_info(f"   - {c.name}")
        else:
            ctx.ui.show_error("⚠️ Tidak ada klip yang dihasilkan.")
            
        self.editor.prune_output_directory(self.config.paths.OUTPUT_DIR)