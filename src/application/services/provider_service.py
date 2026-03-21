from pathlib import Path
from typing import Optional

from src.domain.interfaces import IMediaDownloader, IVideoProcessor, IContentAnalyzer, ICacheManager, ILogger, IRetryHandler
from src.domain.models import VideoSummary, Clip
from src.domain.exceptions import MediaDownloadError, RateLimitError

class ProviderService:
    def __init__(self, downloader: IMediaDownloader, processor: IVideoProcessor, analyzer: IContentAnalyzer, cache_manager: ICacheManager, retry_handler: IRetryHandler, logger: ILogger):
        self.downloader = downloader
        self.processor = processor
        self.analyzer = analyzer
        self.cache_manager = cache_manager
        self.retry_handler = retry_handler
        self.logger = logger

    def get_safe_folder_name(self, url: str) -> Optional[str]:

        return self.downloader.get_safe_title(url)

    def get_transcript(self, url: str, temp_dir: str) -> str:
        try:
            # Gunakan RetryHandler yang sudah dikonfigurasi di Container (Cool Down Policy)
            return self.retry_handler.execute(
                self.downloader.get_transcript, 
                url, 
                output_dir=temp_dir
            )
        except RateLimitError:
            self.logger.error("❌ Rate Limit persisten. Gagal mengambil transkrip setelah retry.")
            return ""
        except MediaDownloadError as e:
            # Error umum download (misal tidak ada sub) tidak di-retry oleh handler ini
            self.logger.warning(f"⚠️ Transkrip tidak ditemukan: {e}. Analisis AI mungkin kurang akurat.")
            return ""

    def prepare_audio_for_analysis(self, url: str, work_dir: Path, filename_prefix: str) -> Path:
        """
        Memastikan file audio WAV yang siap untuk dianalisis tersedia.
        Mengatur alur: Cek Cache -> Unduh (langsung ke WAV) -> Selesai.

        Raises:
            ConnectionError: Jika download gagal.
            IOError: Jika file tidak ditemukan setelah download.
        """
        wav_path = work_dir / f"{filename_prefix}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 10240:
            self.logger.debug(f"♻️ Audio WAV cached: {wav_path.name}")
            return wav_path

        downloaded_audio_path_str = self.downloader.download_audio(url, str(work_dir), filename_prefix)
        downloaded_path = Path(downloaded_audio_path_str)

        if downloaded_path.exists() and downloaded_path.suffix.lower() == '.wav':
            if downloaded_path.resolve() != wav_path.resolve():
                 self.logger.warning(f"File audio diunduh sebagai {downloaded_path.name}, me-rename ke {wav_path.name}")
                 downloaded_path.rename(wav_path)
            return wav_path
        
        raise IOError("Gagal mendapatkan file audio WAV. File tidak ditemukan atau format salah setelah proses unduh.")

    def analyze_video(self, transcript: str, audio_path: str, prompt: str, cache_path: Optional[str] = None, api_key: str = "") -> VideoSummary:
        if cache_path:
            cached_summary = self._load_from_cache(cache_path)
            if cached_summary:
                return cached_summary

        self.logger.info("🧠 Memulai analisis konten dengan AI...")
        summary = self.analyzer.analyze_content(transcript, audio_path, prompt, api_key)

        if cache_path:
            self._save_to_cache(summary, cache_path)

        return summary

    def _load_from_cache(self, path: str) -> Optional[VideoSummary]:
        """Helper internal untuk memuat JSON cache ke Domain Model."""
        data = self.cache_manager.load(path)
        if not data:
            return None
        
        try:
            clips = []
            clips = [Clip.from_dict(c_data) for c_data in data.get('clips', [])]

            return VideoSummary(
                video_title=data.get('video_title', ''),
                audio_energy_profile=data.get('audio_energy_profile', ''),
                clips=clips
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Struktur cache tidak valid: {e}")
            return None

    def _save_to_cache(self, summary: VideoSummary, path: str):
        data = {
            "video_title": summary.video_title,
            "audio_energy_profile": summary.audio_energy_profile,
            "clips": [c.to_dict() for c in summary.clips]
        }
        try:
            self.cache_manager.save(data, path)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyimpan cache analisis: {e}")