from pathlib import Path
from typing import List, Optional, Tuple, Callable

from src.domain.interfaces import IMediaDownloader, IVideoProcessor, IFaceTracker, ITranscriber, ISubtitleWriter, TrackResult, IProgressReporter, ILogger, ISystemHelper
from src.domain.models import Clip

import concurrent.futures

class EditorService:

    def __init__(self, downloader: IMediaDownloader, processor: IVideoProcessor, tracker: IFaceTracker, transcriber: ITranscriber, writer: ISubtitleWriter, system_helper: ISystemHelper, fonts_dir: Path, karaoke_chunk_size: int, logger: ILogger):
        self.downloader = downloader
        self.processor = processor
        self.tracker = tracker
        self.transcriber = transcriber
        self.writer = writer
        self.system_helper = system_helper
        self.fonts_dir = fonts_dir
        self.karaoke_chunk_size = karaoke_chunk_size
        self.logger = logger

    def batch_create_clips(self, clips: List[Clip], source_url: str, output_dir: Path, progress_reporter: Optional[IProgressReporter] = None, cookies_path: Optional[str] = None) -> List[Path]:
        """
        Membuat klip video dari stream URL secara paralel.
        Jumlah workers disesuaikan otomatis berdasarkan jenis encoder (GPU/CPU).
        """
        if self.processor.is_gpu_enabled:
            max_workers = 1 
            self.logger.info("🚀 GPU Encoder terdeteksi: Membatasi proses paralel ke 1 worker untuk stabilitas.")
        else:
            max_workers = 2
            self.logger.info(f"⚙️ CPU Encoder terdeteksi: Menggunakan {max_workers} worker paralel.")

        output_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []
        
        def _process_clip(clip: Clip) -> Optional[Path]:            
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

        self.logger.debug(f"🔄 Memulai pemotongan {len(clips)} klip...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip = {executor.submit(_process_clip, c): c for c in clips}
            
            iterator = concurrent.futures.as_completed(future_to_clip)
            
            # Gunakan progress reporter jika ada
            if progress_reporter:
                iterator = progress_reporter.sequence(iterator, total=len(clips), desc="Cutting Clips", unit="clip")
            
            for future in iterator:
                clip = future_to_clip[future]
                try:
                    path = future.result()
                    if path:
                        created_files.append(path)
                        clip.raw_path = str(path) # Update model domain dengan path fisik
                    else:
                        self.logger.warning(f"⚠️ Gagal membuat klip: {clip.title}")
                except Exception as e:
                    self.logger.error(f"❌ Error pada klip {clip.title}: {e}")

        return sorted(created_files, key=lambda p: p.name)

    def batch_render(
        self,
        tracked_results: List[Tuple[Path, TrackResult]],
        work_dir: Path,
        output_dir: Path,
        progress_reporter: Optional[IProgressReporter] = None
    ) -> List[Path]:
        """
        Merender batch video final secara paralel.
        Mengatur jumlah worker berdasarkan ketersediaan GPU.
        """
        # Tentukan jumlah worker berdasarkan kemampuan hardware (GPU vs CPU)
        if self.processor.is_gpu_enabled:
            max_workers = 1
            self.logger.info("🚀 GPU Encoder terdeteksi: Rendering final dibatasi 1 worker.")
        else:
            max_workers = 2
            self.logger.info(f"⚙️ CPU Encoder terdeteksi: Rendering final menggunakan {max_workers} worker.")

        final_clips = []

        def _process_render(item: Tuple[Path, TrackResult]) -> Optional[Path]:
            original_path, track_res = item
            try:
                sub_path = work_dir / "subs" / f"{original_path.stem}.ass"
                self.generate_subtitles_for_clip(
                    # Jika gagal, method ini akan raise Exception.
                    # Exception akan ditangkap blok try-except ini, dan klip dilewati.
                    str(original_path), 
                    str(sub_path), 
                    work_dir, 
                    self.karaoke_chunk_size, 
                    track_res.width, 
                    track_res.height
                )

                final_out = output_dir / f"final_{original_path.name}"
                
                self.render_final_video(str(track_res.tracked_video), str(original_path), str(sub_path), str(final_out), str(self.fonts_dir))
                return final_out
            except Exception as e:
                self.logger.error(f"❌ Gagal merender klip {original_path.name}: {e}")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_render, item): item for item in tracked_results}
            
            iterator = concurrent.futures.as_completed(futures)
            if progress_reporter:
                iterator = progress_reporter.sequence(iterator, total=len(tracked_results), desc="Rendering Clips", unit="clip")
            
            for future in iterator:
                result = future.result()
                if result:
                    final_clips.append(result)
                    
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
        clip_audio_path: str,
        output_subtitle_path: str,
        cache_dir: Path,
        chunk_size: int,
        video_width: int,
        video_height: int
    ) -> Path:
        """
        Membuat file subtitle .ass untuk satu klip.
        """
        output_path = Path(output_subtitle_path)
        
        if output_path.exists():
            self.logger.debug(f"♻️ Subtitle .ass cached: {output_path.name}")
            return output_path

        try:
            # 1. Transkripsi Audio (Extract words)
            transcription = self.transcriber.transcribe(clip_audio_path)
            
            # 2. Tulis ke format ASS
            self.writer.write_karaoke_subtitles(
                transcription_data=transcription,
                output_path=str(output_path),
                chunk_size=chunk_size,
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