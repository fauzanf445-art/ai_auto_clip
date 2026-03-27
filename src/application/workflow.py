from pathlib import Path
from typing import List, Tuple, Optional

# Import Config & UI
from src.domain.models import Clip, ProjectState, VideoSummary
from src.domain.interfaces import TrackResult, ILogger, IProviderService, IEditorService, IWorkspaceFactory, IAppConfig
from src.domain.exceptions import HSUAIClipError, MediaDownloadError, AnalysisError

from src.application.context import SessionContext

class Workflow:
    def __init__(self, config:IAppConfig, provider: IProviderService, editor: IEditorService, manager_factory: IWorkspaceFactory, logger: ILogger
    ):
        self.config = config
        self.provider = provider
        self.editor = editor
        self.manager_factory = manager_factory
        self.logger = logger

    def _process_url(self, ctx: SessionContext, url: str, work_dir: Path, safe_name: str):
        """Orkestrasi utama: Memproses URL video dari awal hingga akhir. Mengembalikan list klip final."""
        
        # 1. Identifikasi Klip (Manual atau AI)
        # Memuat atau menginisialisasi State
        project_state = self.provider.load_project_state(safe_name)
        project_state.video_source_url = url
        
        clips, context_keywords = self._identify_clips_with_context(ctx, url, work_dir, safe_name)
        if not clips:
            ctx.ui.show_error("Tidak ada klip yang ditemukan atau dibuat.")
            return []

        # 2. Ekstraksi (Download & Potong)
        raw_clip_paths = self._cut_raw_clips(ctx, clips, url, work_dir, project_state, safe_name)
        if not raw_clip_paths:
            raise MediaDownloadError("Gagal memotong klip mentah sama sekali. Periksa koneksi atau log.")
            
        # 3. Pemrosesan (Tracking & Rendering)
        # Mengirim project_state dan clips untuk fitur resume & update status
        tracked_results = self._track_clips(ctx, raw_clip_paths, work_dir, project_state, clips, safe_name)
        
        # Perkaya prompt config global dengan judul video spesifik
        final_clips = self._render_final_clips(ctx, tracked_results, work_dir, safe_name, context_keywords, project_state, clips)

        # 4. Finalisasi (UI & Cleanup)
        self._finalize_processing(ctx, safe_name, final_clips)

        return final_clips

    def run(self, ctx: SessionContext, url: str) -> List[Path]:
        """Menjalankan pipeline lengkap."""
        generated_clips: List[Path] = []
        try:
            ctx.ui.show_info("🚀 [STEP] Inisialisasi...")
            video_metadata = self.provider.get_safe_folder_name(url)
            if not video_metadata:
                raise MediaDownloadError("Gagal mengambil metadata video (Judul/ID). Pastikan URL valid atau cookies terkonfigurasi.")

            with self.manager_factory.create(video_metadata) as (safe_name, work_dir):                
                ctx.ui.show_info(f"   -> Working Directory: {work_dir}")
                generated_clips = self._process_url(ctx, url, work_dir, safe_name)

        except HSUAIClipError as e:
            self.logger.error(f"Pipeline Error: {e}")
            ctx.ui.show_error(str(e))
        except Exception as e:
            self.logger.error("Orchestrator Error", exc_info=True)
            ctx.ui.show_error(str(e))
            
        return generated_clips

    def _identify_clips_with_context(self, ctx: SessionContext, url: str, work_dir: Path, safe_name: str) -> Tuple[List[Clip], str]:
        """
        Menentukan sumber klip dan mengembalikan konteks judul video.
        Returns: (List[Clip], context_keywords)
        """
        ctx.ui.show_info("🚀 [STEP] Analisis Konten...")
        
        # Coba mode manual terlebih dahulu
        manual_clips = self._try_get_manual_clips()
        if manual_clips:
            # Untuk manual, kita coba ambil judul dari provider, atau gunakan placeholder
            title = self.provider.get_safe_folder_name(url) or "Manual Video"
            return manual_clips, title
            
        # Fallback ke mode AI
        clips, context_keywords = self._perform_ai_analysis(ctx, url, work_dir, safe_name)
        return clips, context_keywords

    def _try_get_manual_clips(self) -> Optional[List[Clip]]:
        """Mencoba mendapatkan input timestamp dari user."""
        # Logika interaktif ini sebaiknya di Controller, tapi untuk transisi:
        # Kita lewati manual input jika UI tidak interaktif atau implementasi sederhana
        # self.ui.get_input("Masukkan timestamp manual") -> Logic parsing pindah sini
        # Untuk refactoring tahap ini, kita matikan dulu fitur manual via Workflow
        # agar fokus pada pembersihan UI.
        return None

    def _perform_ai_analysis(self, ctx: SessionContext, url: str, work_dir: Path, safe_name: str) -> Tuple[List[Clip], str]:
        """Menjalankan analisis AI lengkap (Transkrip + Audio + Gemini)."""
        ctx.ui.show_info("   -> Mode AI: Menganalisis video untuk klip potensial...")
        
        # Membuat subfolder 'source' untuk menyimpan file mentah dan cache analisis
        source_dir = work_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        audio_wav_path = self.provider.get_audio_for_analysis(url, source_dir, "full_audio")
        
        if not audio_wav_path:
            raise MediaDownloadError("Gagal menyiapkan audio untuk analisis.")
        
        prompt_path = self.config.paths.prompt_file
        
        # Hybrid Lazy Init: Minta API Key jika belum tersedia di Context
        if not ctx.api_key:
            self.logger.warning("🔑 Gemini API Key tidak ditemukan.")
            ctx.api_key = ctx.ui.get_secure_input("Masukkan Gemini API Key untuk melanjutkan analisis")

        summary = self.provider.analyze_video(
            url=url,
            temp_dir=source_dir,
            filename_prefix="full_audio",
            cache_path=safe_name,
            prompt_path=str(prompt_path),
            audio_path=str(audio_wav_path),
            api_key=ctx.api_key  # Passing API Key dari Context
        )
        ctx.ui.show_info(f"   -> AI menemukan {len(summary.clips)} klip potensial.")
        return summary.clips, summary.context_keywords

    def _cut_raw_clips(self, ctx: SessionContext, clips: List[Clip], url: str, work_dir: Path, project_state: ProjectState, safe_name: str) -> List[Path]:
        """
        Memotong klip mentah dari stream video. 
        Menggunakan strategi 'Direct Segment Download' dengan pemaksaan CFR.
        """
        ctx.ui.show_info("🚀 [STEP] Downloading & CFR Conversion...")
        # Note: Kita menggunakan URL asli YouTube dan yt-dlp downloader
        raw_clips_dir = work_dir / "raw_clips"

        # --- RESUME LOGIC ---
        clips_to_download = []
        cached_paths = []

        for clip in clips:
            state = project_state.get_clip_state(clip.id)
            # Cek apakah status valid (DOWNLOADED atau tahap selanjutnya) dan file fisik ada
            if state.status in ["DOWNLOADED", "TRACKED", "COMPLETED"] and state.raw_path and Path(state.raw_path).exists():
                 self.logger.info(f"♻️  Skip download (Resume): {clip.title}")
                 cached_paths.append(Path(state.raw_path))
            else:
                clips_to_download.append(clip)
        
        if not clips_to_download:
            ctx.ui.show_info(f"   -> Semua {len(clips)} klip sudah tersedia (Resume).")
            return cached_paths

        # Proses klip yang belum ada
        created_clip_paths = self.editor.batch_create_clips(
            clips=clips_to_download,
            source_url=url,
            output_dir=raw_clips_dir,
            progress_reporter=ctx.progress_reporter
        )
        
        # Update Project State: Handle Success & Failure
        created_stems = {p.stem for p in created_clip_paths}
        
        for clip in clips_to_download:
            if clip.safe_filename in created_stems:
                # Success
                path = next((p for p in created_clip_paths if p.stem == clip.safe_filename), None)
                if path:
                    project_state.update_state(clip.id, raw_path=str(path), status="DOWNLOADED")
            else:
                # Failure
                self.logger.warning(f"❌ Klip gagal didownload/dipotong: {clip.title}")
                project_state.update_state(clip.id, status="FAILED")
        
        self.provider.save_project_state(safe_name, project_state)
        ctx.ui.show_info(f"   -> {len(created_clip_paths)} klip baru, {len(cached_paths)} dari cache.")
        
        return cached_paths + created_clip_paths

    def _track_clips(self, ctx: SessionContext, raw_clip_paths: List[Path], work_dir: Path, project_state: ProjectState, clips: List[Clip], safe_name: str) -> List[Tuple[Path, TrackResult]]:
        """Menjalankan motion tracking pada setiap klip mentah."""
        ctx.ui.show_info("🚀 [STEP] Motion Tracking (MediaPipe)...")
        tracked_dir = work_dir / "tracked_clips"
        tracked_results: List[Tuple[Path, TrackResult]] = []
        
        # Identifikasi klip yang perlu ditrack vs resume
        paths_to_process = []
        
        for path in raw_clip_paths:
            # Cari klip yang sesuai dengan path ini
            matched_clip = next((c for c in clips if c.safe_filename == path.stem), None)
            if not matched_clip:
                continue

            state = project_state.get_clip_state(matched_clip.id)
            
            # Cek Resume: Apakah sudah TRACKED/COMPLETED dan file tracked ada?
            if state.status in ["TRACKED", "COMPLETED"] and state.tracked_path and Path(state.tracked_path).exists():
                self.logger.debug(f"♻️  Skip tracking (Resume): {path.name}")
                # Kita perlu me-reconstruct TrackResult secara manual atau memuatnya jika disimpan (disini kita simulative)
                # Karena TrackResult butuh width/height, idealnya kita simpan di JSON state.
                # Untuk simplifikasi saat ini, kita anggap dimensi standar 9:16 (misal 608x1080) atau baca dari file.
                # Solusi Robust: Track ulang cepat atau baca metadata video. Di sini kita proses ulang jika data dimensi hilang,
                # atau untuk amannya, kita asumsikan tracking ulang (MediaPipe) lebih cepat daripada error.
                # Namun untuk mematuhi prompt 'Resume', mari kita asumsikan kita proses ulang jika ragu, 
                # TAPI jika file tracked ada, kita gunakan file itu.
                
                # TODO: Idealnya ProjectState menyimpan dimensi video hasil tracking.
                # Fallback: Kita gunakan file yang sudah ada, tapi baca dimensinya via OpenCV/FFprobe.
                # Disini kita skip logic kompleks dan masukkan ke antrian process jika dimensi tidak diketahui,
                # ATAU kita anggap tracking MediaPipe cukup cepat untuk dijalankan ulang pada file cached?
                # Sesuai prompt "Resume", kita coba skip jika file output ada.
                
                # Karena EditorService.track_subject melakukan crop, kita sebenarnya butuh TrackResult untuk tahap rendering (subtitle placement).
                # Tanpa width/height yang akurat, subtitle bisa berantakan.
                # KEPUTUSAN: Jalankan tracking ulang (idempotent) untuk mendapatkan TrackResult, 
                # tapi EditorService akan skip processing berat jika output file sudah ada (cache check di dalam track_subject).
                paths_to_process.append((path, matched_clip))
            else:
                paths_to_process.append((path, matched_clip))
        
        # Outer progress bar untuk setiap klip
        if ctx.progress_reporter:
            iterator = ctx.progress_reporter.sequence(paths_to_process, desc="Overall Tracking", unit="clip")
        else:
            iterator = paths_to_process

        for clip_path, clip in iterator:
            self.logger.debug(f"   -> Tracking klip: {clip_path.name}")
            output_tracked = tracked_dir / f"{clip_path.name}"
            
            try:
                # Eksekusi Sekuensial: Mencegah OOM dengan memproses satu per satu
                # EditorService.track_subject memiliki logic internal untuk skip jika output ada
                result = self.editor.track_subject(str(clip_path), str(output_tracked),progress_reporter=ctx.progress_reporter)
                tracked_results.append((clip_path, result))
                
                # Update Project State
                project_state.update_state(clip.id, tracked_path=str(output_tracked), status="TRACKED")
                self.provider.save_project_state(safe_name, project_state)
                
            except Exception as e:
                self.logger.error(f"❌ Gagal tracking klip {clip_path.name}: {e}")
                ctx.ui.show_info(f"   -> ⚠️ Skip klip {clip_path.name} karena error tracking.")
                project_state.update_state(clip.id, status="FAILED")
                self.provider.save_project_state(safe_name, project_state)

        return tracked_results

    def _render_final_clips(self, ctx: SessionContext, tracked_results: List[Tuple[Path, TrackResult]], work_dir: Path, safe_name: str, context_keywords: str, project_state: ProjectState, clips: List[Clip]) -> List[Path]:
        """Membuat subtitle dan merender video final."""
        ctx.ui.show_info("🚀 [STEP] Captioning & Rendering Final...")
        
        final_paths = self.editor.batch_render(
            tracked_results=tracked_results,
            clips=clips,
            work_dir=work_dir,
            output_dir=self.config.paths.output_dir / safe_name,
            progress_reporter=ctx.progress_reporter
        )
        
        final_stems = {p.stem for p in final_paths}

        # Update State untuk setiap file yang berhasil dirender
        for path in final_paths:
            for clip in clips:
                if clip.safe_filename == path.stem:
                     project_state.update_state(clip.id, final_path=str(path), status="COMPLETED")
        
        # Identifikasi kegagalan render
        for raw_path, _ in tracked_results:
            if raw_path.stem not in final_stems:
                 matched_clip = next((c for c in clips if c.safe_filename == raw_path.stem), None)
                 if matched_clip:
                     self.logger.warning(f"❌ Klip gagal render final: {matched_clip.title}")
                     project_state.update_state(matched_clip.id, status="FAILED")

        self.provider.save_project_state(safe_name, project_state)
        return final_paths

    def _finalize_processing(self, ctx: SessionContext, safe_name: str, final_clips: List[Path]):
        """Menampilkan hasil sukses dan membersihkan output lama."""
        output_folder = self.config.paths.output_dir / safe_name
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
            
        self.editor.prune_output_directory(self.config.paths.output_dir)