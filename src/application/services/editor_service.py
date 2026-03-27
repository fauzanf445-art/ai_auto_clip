from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict

from src.domain.interfaces import IYoutubeAdapter, IFfmpegAdapter, IMediapipeAdater, ISubtitleWriter, TrackResult, IProgressReporter, ILogger, ISystemHelper, IAppConfig
from src.domain.models import Clip, TranscriptionWord, TranscriptionSegment

class EditorService:

    def __init__(self, config: IAppConfig, downloader: IYoutubeAdapter, processor: IFfmpegAdapter, tracker: IMediapipeAdater, writer: ISubtitleWriter, system_helper: ISystemHelper, fonts_dir: Path, logger: ILogger):
        self.config = config
        self.downloader = downloader
        self.processor = processor
        self.tracker = tracker
        self.writer = writer
        self.system_helper = system_helper
        self.fonts_dir = fonts_dir
        self.logger = logger

    def batch_create_clips(self, clips: List[Clip], source_url: str, output_dir: Path, progress_reporter: Optional[IProgressReporter] = None, cookies_path: Optional[str] = None) -> List[Path]:
        """
        Membuat klip video dari stream URL secara paralel.
        Jumlah workers disesuaikan otomatis berdasarkan jenis encoder (GPU/CPU).
        """
        self.logger.info("🎬 Memulai pemotongan klip secara sekuensial...")

        output_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []
        
        # Helper internal untuk memproses satu klip
        def _process_single_clip(clip: Clip) -> Optional[Path]:
            filename = f"{clip.safe_filename}.mp4"
            output_path = output_dir / filename
            
            # Cek cache
            if output_path.exists() and output_path.stat().st_size > 1024:
                self.logger.debug(f"♻️  Klip cached: {filename}")
                return output_path

            self.downloader.download_video_section(
                url=source_url,
                start=clip.start_time,
                end=clip.end_time,
                output_path=str(output_path)
            )
            
            if output_path.exists():
                # File sudah dipastikan CFR oleh adapter
                return output_path
            
            return None

        # Gunakan progress reporter jika ada
        iterator = clips
        if progress_reporter:
            iterator = progress_reporter.sequence(clips, total=len(clips), desc="Cutting Clips", unit="clip")

        for clip in iterator:
            try:
                path = _process_single_clip(clip)
                if path:
                    created_files.append(path)
                else:
                    self.logger.warning(f"⚠️ Gagal membuat klip: {clip.title}")
            except Exception as e:
                self.logger.error(f"❌ Error pada klip {clip.title}: {e}")

        return sorted(created_files, key=lambda p: p.name)

    def batch_render(
        self,
        tracked_results: List[Tuple[Path, TrackResult]],
        clips: List[Clip],
        work_dir: Path,
        output_dir: Path,
        progress_reporter: Optional[IProgressReporter] = None
    ) -> List[Path]:
        """
        Merender batch video final secara paralel.
        Mengatur jumlah worker berdasarkan ketersediaan GPU.
        """
        self.logger.info("🎨 Memulai rendering final secara sekuensial...")

        # Buat lookup map untuk mencari Clip berdasarkan nama file
        clip_map = {c.safe_filename: c for c in clips}
        final_clips = []

        def _process_single_render(item: Tuple[Path, TrackResult]) -> Optional[Path]:
            original_path, track_res = item
            
            # Ambil data klip untuk mendapatkan kata-kata transkrip
            clip = clip_map.get(original_path.stem)
            if not clip:
                self.logger.warning(f"⚠️ Metadata klip tidak ditemukan untuk {original_path.name}. Skip subtitle.")
                return None

            try:
                sub_path = work_dir / "subs" / f"{original_path.stem}.ass"
                self.generate_subtitles_for_clip(
                    words=clip.words,
                    clip_start_time=clip.start_time,
                    output_subtitle_path=str(sub_path),
                    video_width=track_res.width,
                    video_height=track_res.height
                )

                final_out = output_dir / f"{original_path.name}"
                
                self.render_final_video(str(track_res.tracked_video), str(original_path), str(sub_path), str(final_out), str(self.fonts_dir))
                return final_out
            except Exception as e:
                self.logger.error(f"❌ Gagal merender klip {original_path.name}: {e}")
            return None

        # Gunakan progress reporter jika ada
        iterator = tracked_results
        if progress_reporter:
            iterator = progress_reporter.sequence(tracked_results, total=len(tracked_results), desc="Rendering Clips", unit="clip")

        for item in iterator:
            try:
                result = _process_single_render(item)
                if result:
                    final_clips.append(result)
            except Exception as e:
                self.logger.error(f"❌ Fatal error rendering item: {e}")
                    
        return sorted(final_clips, key=lambda p: p.name)

    def track_subject(self, input_path: str, output_path: str, progress_reporter: Optional[IProgressReporter] = None) -> TrackResult:
        """
        Menjalankan motion tracking pada video input.
        """
        # Setup progress bar
        pbar = None
        _internal_progress_callback: Optional[Callable[[int, int], None]] = None

        if progress_reporter:
            pbar = progress_reporter.manual(total=100, desc="   -> Frames", unit="frame", leave=False)

            def update_progress(current_frame: int, total_frames: int):
                if pbar.total != total_frames:
                    pbar.total = total_frames
                pbar.update(current_frame - pbar.n)
            _internal_progress_callback = update_progress

        try:
            return self.tracker.track_and_crop(input_path, output_path, _internal_progress_callback)
        finally:
            if pbar:
                pbar.close()

    def render_final_video(self, video_path: str, audio_path: str, subtitle_path: Optional[str], output_path: str, fonts_dir: Optional[str] = None) -> None:
        """
        Merender video final dengan subtitle dan audio asli.
        """
        self.processor.render_final(video_path, audio_path, subtitle_path, output_path, fonts_dir)

    def generate_subtitles_for_clip(
        self,
        words: List[TranscriptionWord],
        clip_start_time: float,
        output_subtitle_path: str,
        video_width: int,
        video_height: int
    ) -> Path:
        """
        Membuat file subtitle .ass untuk satu klip menggunakan data kata yang sudah ada.
        Tidak melakukan transkripsi ulang, hanya menggeser timestamp (Shifting).
        """
        output_path = Path(output_subtitle_path)
        
        if output_path.exists():
            self.logger.debug(f"♻️ Subtitle .ass cached: {output_path.name}")
            return output_path

        try:
            if not words:
                self.logger.warning(f"⚠️ Tidak ada data kata untuk subtitle: {output_path.name}")
                # Buat file kosong atau handle sesuai kebutuhan
                return output_path

            # 1. Shift Timestamp
            # Timestamp di 'words' adalah relatif terhadap video ASLI.
            # Kita harus mengubahnya menjadi relatif terhadap awal KLIP (mulai dari 0).
            shifted_words = []
            for w in words:
                new_start = max(0.0, w.start - clip_start_time)
                new_end = max(0.0, w.end - clip_start_time)
                shifted_words.append(TranscriptionWord(
                    word=w.word, start=new_start, end=new_end, probability=w.probability
                ))
            
            # 2. Bungkus dalam Segment (Adapter Pattern untuk ISubtitleWriter)
            # Writer mengharapkan List[TranscriptionSegment]
            dummy_segment = TranscriptionSegment(
                start=shifted_words[0].start,
                end=shifted_words[-1].end,
                text="", # Tidak digunakan oleh karaoke writer
                words=shifted_words
            )
            
            self.writer.write_ass_sub_style(
                transcription_data=[dummy_segment],
                output_path=str(output_path),
                play_res_x=video_width,
                play_res_y=video_height
            )
            return output_path
        except Exception as e:
            # Re-raise agar batch_render tahu ini gagal dan bisa skip/handle klip ini
            raise Exception(f"Gagal membuat subtitle: {e}") from e

    def prune_output_directory(self, output_dir: Path, max_files: int = 10, max_size_mb: int = 500):
        """
        Memangkas folder output jika melebihi batas ukuran atau jumlah file.
        Menghapus file video tertua (berdasarkan waktu modifikasi) terlebih dahulu.
        """
        self.system_helper.prune_directory(
            directory=output_dir,
            max_files=max_files,
            max_size_mb=max_size_mb,
            file_prefix="final_",
            extensions=('.mp4', '.mov')
        )