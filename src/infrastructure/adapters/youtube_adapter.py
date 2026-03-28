import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union

from src.application.context import SessionContext
from src.domain.interfaces import IYoutubeAdapter, ILogger
from src.domain.exceptions import MediaDownloadError, RateLimitError

class YouTubeAdapter(IYoutubeAdapter):
    """
    Implementasi IMediaDownloader menggunakan yt-dlp.
    Menangani interaksi dengan YouTube: Metadata, Stream URL, Audio Download, dan Cookies.
    """

    def __init__(self, yt_dlp_path: str, logger: ILogger, node_path: Optional[str] = None, cookies_path: Optional[Union[str, Path]] = None):
        self.cookies_path = cookies_path
        self._info_cache: Dict[str, Any] = {}
        self.bin_path = yt_dlp_path
        self.logger = logger

        self.base_cli_args = [
            '--force-ipv4',
            '--verbose',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--replace-in-metadata', 'title', '[<>:"/\\\\|?*]', '',
            '--replace-in-metadata', 'title', '[\\s\\.]+', '_',
            '--replace-in-metadata', 'title', '[^a-zA-Z0-9_]', '',
            '--retries', '10',  
            '--fragment-retries', '10',
            '--retry-sleep', 'exp=1:20'
        ]
        if self.cookies_path and Path(self.cookies_path).exists():
            self.base_cli_args.extend(['--cookies', str(self.cookies_path)])

        if node_path:
            self.base_cli_args.extend(['--js-runtimes', f'node:{node_path}'])
        else:
            self.logger.warning("⚠️ Executable 'node' tidak ditemukan/diberikan. Beberapa video YouTube mungkin gagal diunduh.")

    def _execute_command(self, ctx: SessionContext, cmd: list, timeout: int = 300, require_stdout: bool = False) -> str:
        """Helper internal untuk menjalankan command subprocess dengan handling standar."""
        try:
            kwargs = {
                'text': True,
                'encoding': 'utf-8',
                'check': True,
                'timeout': timeout
            }
            
            if require_stdout:
                kwargs['capture_output'] = True
            else:
                kwargs['stdout'] = subprocess.DEVNULL
                kwargs['stderr'] = subprocess.PIPE

            result = subprocess.run(cmd, **kwargs)
            return result.stdout.strip() if require_stdout else ""

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
            if isinstance(error_msg, bytes): # Fallback jika text=False (meski kita set True)
                error_msg = error_msg.decode('utf-8', errors='ignore')
            
            if "HTTP Error 429" in error_msg or "Too Many Requests" in error_msg:
                raise RateLimitError(f"YouTube Rate Limit detected (IP Blocked): {error_msg}") from e
            raise MediaDownloadError(f"Process Failed: {error_msg}") from e
        except subprocess.TimeoutExpired as e:
            raise MediaDownloadError(f"Process timeout after {e.timeout}s") from e

    def get_safe_title(self, ctx: SessionContext, url: str) -> str:
        
        """
        Mengambil judul video yang aman digunakan sebagai nama folder.
        """
        if url in self._info_cache:
            return self._info_cache[url]

        cmd = [
            self.bin_path,
            '--get-title',
            '--no-playlist',
        ] + self.base_cli_args + [url]

        try:
            ctx.logger.debug(f"   -> Getting safe title: {url}")
            safe_title = self._execute_command(ctx, cmd, timeout=60, require_stdout=True)
            self._info_cache[url] = safe_title
            return safe_title
        except MediaDownloadError as e:
            raise MediaDownloadError(f"Gagal mendapatkan judul video: {e}") from e

    def download_audio(self, ctx: SessionContext, url: str, output_dir: str, filename_prefix: str) -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        output_template = str(out_path / f"{filename_prefix}.%(ext)s")
        
        cmd: list[str] = [
            self.bin_path,
            '-x',
            '--audio-format', 'wav',
            '--output', output_template,
            url
        ] + self.base_cli_args

        ctx.logger.debug(f"🎵 Mengunduh audio via CLI: {filename_prefix}...")
        ctx.logger.info(f"⬇️ Mengunduh audio dari {url}...")
        self._execute_command(ctx, cmd, timeout=300)

        for file_path in out_path.glob(f"{filename_prefix}.*"):
            if file_path.suffix not in ['.part', '.ytdl']:
                return str(file_path)

        raise MediaDownloadError(f"File output audio tidak ditemukan setelah download.")

    def download_video_section(self, ctx: SessionContext, url: str, start: float, end: float, output_path: str) -> None:
        """
        Mengunduh bagian spesifik dari video YouTube menggunakan yt-dlp.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        section_arg = f"*{start:.2f}-{end:.2f}"
        
        # Post-processor args to force CFR 30fps, AAC audio, and normalize video
        # Note: We apply this to 'ffmpeg' postprocessor
        pp_args: str = "ffmpeg:-r 30 -vsync cfr -c:v libx264 -preset ultrafast -c:a aac -b:a 192k"

        cmd = [
            self.bin_path,
            '--download-sections', section_arg,
            '--force-keyframes-at-cuts',
            '--postprocessor-args', pp_args,
            '--concurrent-fragments', '4',
        ] + self.base_cli_args + [
            '-f', 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
            '-o', output_path,
            url
        ]

        ctx.logger.debug(f"✂️ Downloading CFR Segment: {Path(output_path).name} ({start}-{end}s)")
        self._execute_command(ctx, cmd, timeout=300)
        
