import subprocess
import os
from pathlib import Path
from typing import List, Optional, Callable, Tuple

from src.domain.interfaces import IVideoProcessor, ILogger
from src.domain.exceptions import VideoProcessingError

class FFmpegAdapter(IVideoProcessor):
    """
    Implementasi IVideoProcessor menggunakan FFmpeg CLI.
    Menangani pemotongan, konversi, dan rendering video dengan deteksi hardware acceleration.
    """
    
    # Konstanta teknis
    CLIP_END_PADDING_SECONDS = 0.15
    SEEK_BUFFER_SECONDS = 5.0
    
    # Argumen audio standar
    AAC_AUDIO_ARGS = [
        '-c:a', 'aac',
        '-ar', '44100',
        '-b:a', '192k'
    ]
    
    # Argumen video CPU sebagai fallback
    CPU_VIDEO_ARGS = ['-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
    
    # Definisi Argumen Encoder (Centralized)
    NVENC_ARGS = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '24', '-rc', 'vbr', '-tune', 'hq', '-pix_fmt', 'yuv420p']
    QSV_ARGS = ['-c:v', 'h264_qsv', '-global_quality', '23', '-preset', 'veryfast', '-pix_fmt', 'nv12']
    AMF_ARGS = ['-c:v', 'h264_amf', '-quality', '2', '-pix_fmt', 'yuv420p']
    VIDEOTOOLBOX_ARGS = ['-c:v', 'h264_videotoolbox', '-b:v', '4M', '-pix_fmt', 'yuv420p']

    def __init__(self, ffmpeg_path: str, ffprobe_path: str, logger: ILogger, encoder_preference: Optional[str] = None):
        self.bin_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.encoder_preference = encoder_preference
        self.logger = logger
        self._video_args: List[str] = []
        self._common_args: List[str] = []
        self._codec_args: List[str] = []
        self._setup_encoder_definitions()

    @property
    def is_gpu_enabled(self) -> bool:
        """Mengembalikan True jika encoder yang aktif bukan CPU default."""
        if not self._codec_args:
            self.initialize()
        return self._video_args != self.CPU_VIDEO_ARGS

    def _setup_encoder_definitions(self):
        """Menginisialisasi definisi encoder untuk prioritas dan lookup."""
        # Daftar prioritas untuk deteksi otomatis (Tuple: nama_ffmpeg, friendly_name, args)
        self.HARDWARE_ENCODERS = [
            ('h264_nvenc', "NVIDIA NVENC", self.NVENC_ARGS),
            ('h264_qsv', "Intel QuickSync (QSV)", self.QSV_ARGS),
            ('h264_amf', "AMD AMF", self.AMF_ARGS),
            ('h264_videotoolbox', "Apple VideoToolbox", self.VIDEOTOOLBOX_ARGS)
        ]

        # Mapping untuk preferensi manual (Config/Env -> Friendly Name, Args)
        self.ENCODER_PRESETS = {
            'cpu': ('CPU (libx264)', self.CPU_VIDEO_ARGS),
            'nvenc': ("NVIDIA NVENC", self.NVENC_ARGS),
            'qsv': ("Intel QuickSync (QSV)", self.QSV_ARGS),
            'amf': ("AMD AMF", self.AMF_ARGS),
            'videotoolbox': ("Apple VideoToolbox", self.VIDEOTOOLBOX_ARGS)
        }

    def get_video_duration(self, path: str) -> Optional[float]:
        """Validasi durasi video menggunakan ffprobe untuk memastikan file valid."""
        try:
            cmd = [
                self.ffprobe_path, 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Gagal memvalidasi durasi {path}: {e}")
            return None

    @staticmethod
    def _escape_ffmpeg_path(path_str: str) -> str:
        """
        Meng-escape path untuk digunakan di dalam filter complex FFmpeg.
        Ini penting untuk Windows (meng-handle C: dan path dengan karakter spesial.
        """
        p = Path(path_str).resolve().as_posix()
        # Di Windows, C:\path menjadi C:/path, lalu di-escape menjadi C\:/path
        return p.replace(':', '\\:')

    def _is_encoder_functional(self, encoder_name: str, test_args: List[str]) -> bool:
        """
        Menjalankan verifikasi aktif untuk sebuah encoder dengan command FFmpeg singkat.
        """
        self.logger.debug(f"   -> Memverifikasi fungsionalitas encoder: {encoder_name}...")
        cmd = [
            self.bin_path, '-nostats', '-y',
            '-f', 'lavfi', '-i', 'color=s=64x64:rate=30', # Input virtual kecil
            '-t', '0.1', # Durasi sangat pendek
        ]
        cmd.extend(test_args)
        cmd.extend(['-f', 'null', '-']) # Output ke null device

        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if process.returncode == 0:
                self.logger.debug(f"   ✅ Verifikasi {encoder_name} berhasil.")
                return True
            else:
                self.logger.debug(f"   ⚠️ Verifikasi {encoder_name} gagal. FFmpeg stderr:\n{process.stderr}")
                return False
        except FileNotFoundError:
            # This happens if ffmpeg itself is not found
            self.logger.error("❌ FFmpeg tidak ditemukan. Pastikan sudah terinstall dan ada di PATH sistem atau di folder 'bin'.")
            raise
        except Exception as e:
            self.logger.warning(f"   ⚠️ Exception saat verifikasi {encoder_name}: {e}")
            return False

    def _determine_best_encoder(self) -> Tuple[str, List[str]]:
        # Menggunakan definisi terpusat
        for name, friendly_name, args in self.HARDWARE_ENCODERS:
            if self._is_encoder_functional(name, args):
                self.logger.info(f"🚀 FFmpeg Adapter: Menggunakan akselerasi hardware {friendly_name}.")
                return friendly_name, args

        self.logger.info("⚙️ FFmpeg Adapter: Tidak ada akselerasi hardware fungsional yang terdeteksi. Menggunakan CPU (libx264).")
        return "CPU", self.CPU_VIDEO_ARGS

    def initialize(self):
        if self._codec_args:
            return

        video_args_determined = False

        # Prioritas 1: Environment Variable (FFMPEG_ENCODER)
        env_pref = os.getenv('FFMPEG_ENCODER', '').lower().strip()
        if env_pref and env_pref in self.ENCODER_PRESETS:
            name, args = self.ENCODER_PRESETS[env_pref]
            self.logger.info(f"🚀 FFmpeg Adapter: Menggunakan encoder via Environment Variable ({name}).")
            self._video_args = args
            video_args_determined = True

        # Prioritas 2: Config Preference
        if not video_args_determined and self.encoder_preference:
            conf_pref = self.encoder_preference.lower().strip()
            if conf_pref in self.ENCODER_PRESETS:
                name, args = self.ENCODER_PRESETS[conf_pref]
                self.logger.info(f"🚀 FFmpeg Adapter: Menggunakan encoder via Config ({name}).")
                self._video_args = args
                video_args_determined = True

        # Prioritas 3 (sebelumnya 4): Deteksi Otomatis (Fallback)
        if not video_args_determined:
            friendly_name, self._video_args = self._determine_best_encoder()
        
        self._common_args = [
            '-r', '30', '-vsync', '1',
            '-avoid_negative_ts', 'make_zero',
            '-fflags', '+genpts+igndts',
            '-map_metadata', '0',
            '-threads', '0'
        ]
        
        self._codec_args = self._common_args + self._video_args + self.AAC_AUDIO_ARGS
        self.logger.debug(f"FFmpeg codec args initialized: {' '.join(self._codec_args)}")

    def _get_codec_args(self) -> List[str]:
        if not self._codec_args:        
            self.logger.debug("Lazy initialization: Detecting FFmpeg hardware support...")
            self.initialize()
        return self._codec_args

    def _get_cpu_codec_args(self) -> List[str]:
        """Mengembalikan argumen codec khusus untuk fallback CPU."""
        return self._common_args + self.CPU_VIDEO_ARGS + self.AAC_AUDIO_ARGS

    def _run_command(self, cmd: List[str], description: str, log_error: bool = True) -> None:
        """Helper untuk menjalankan subprocess dengan logging."""
        try:
            # Hapus argumen -nostats agar log bersih, karena stderr akan ditangkap
            cmd = [c for c in cmd if c != '-nostats']
            
            self.logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600 # Generous timeout for rendering
            )
            
            if process.returncode != 0:
                error_tail = "\n".join(process.stderr.splitlines()[-20:])
                if log_error:
                    self.logger.error(f"❌ FFmpeg Error ({description}):\n... [Log dipotong] ...\n{error_tail}")
                raise VideoProcessingError(f"FFmpeg gagal ({description}): {error_tail}")

        except Exception as e:
            if isinstance(e, VideoProcessingError):
                raise e
            raise VideoProcessingError(f"Exception sistem saat menjalankan FFmpeg ({description}): {e}")

    def _run_with_fallback(self, build_cmd_func: Callable[[List[str]], List[str]], description: str) -> None:
        """
        Menjalankan command dengan mekanisme fallback ke CPU jika gagal.
        Tidak mengubah state instance secara permanen.
        """
        try:
            # Percobaan 1: Menggunakan argumen yang sudah di-cache (GPU atau CPU default)
            codec_args = self._get_codec_args()
            cmd = build_cmd_func(codec_args)
            self._run_command(cmd, description, log_error=False)
            return

        except VideoProcessingError as e:
            # Jika gagal, dan kita tidak sedang dalam mode CPU, coba fallback
            if self._video_args == self.CPU_VIDEO_ARGS:
                # Sudah CPU, tidak ada harapan lagi
                raise e

            self.logger.warning(f"⚠️ Deteksi kegagalan pada {description}. Mencoba fallback ke CPU... Error: {e}")
            
            # Percobaan 2: Menggunakan argumen CPU secara eksplisit
            cpu_codec_args = self._get_cpu_codec_args()
            cmd_fallback = build_cmd_func(cpu_codec_args)
            
            self._run_command(cmd_fallback, f"{description} (CPU Fallback)")

    def render_final(self, video_path: str, audio_path: str, subtitle_path: Optional[str], output_path: str, fonts_dir: Optional[str] = None) -> None:
        """
        Merender hasil akhir: Video Tracked + Audio Asli + Subtitle (Burn-in).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        def build_cmd(codec_args: List[str]) -> List[str]:
            # Filter Chain Construction
            filter_chain = "[0:v]null[v_out]"
            
            if subtitle_path and os.path.exists(subtitle_path):
                esc_sub = self._escape_ffmpeg_path(subtitle_path)
                
                fonts_opt = ""
                if fonts_dir and os.path.exists(fonts_dir):
                    esc_fonts = self._escape_ffmpeg_path(fonts_dir)
                    fonts_opt = f":fontsdir='{esc_fonts}'"

                filter_chain = f"[0:v]ass='{esc_sub}'{fonts_opt}[v_out]"

            cmd = [
                self.bin_path, '-nostats', '-y',
                '-i', video_path, # Input 0
                '-i', audio_path, # Input 1
                '-filter_complex', filter_chain,
                '-map', '[v_out]', '-map', '1:a:0',
                '-shortest'
            ]
            
            cmd.extend(codec_args)
            cmd.append(output_path)
            return cmd

        self._run_with_fallback(build_cmd, f"Render Final: {Path(output_path).name}")
