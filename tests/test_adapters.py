import unittest
from unittest.mock import patch, MagicMock, ANY, mock_open
from pathlib import Path
import os
import subprocess

from src.domain.exceptions import ExecutableNotFoundError
from src.infrastructure.adapters.youtube_adapter import YouTubeAdapter
from src.infrastructure.adapters.ffmpeg_adapter import FFmpegAdapter
from src.infrastructure.adapters.whisper_adapter import WhisperAdapter
from src.infrastructure.adapters.subtitle_writer import AssSubtitleWriter
from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.infrastructure.adapters.mediapipe_adapter import MediaPipeAdapter
from src.domain.interfaces import ILogger, IRetryHandler, IFileDownloader, ITextProcessor
from src.domain.models import TranscriptionSegment, TranscriptionWord
from src.domain.exceptions import MediaDownloadError, VideoProcessingError, AnalysisError, TranscriptionError

class TestYouTubeAdapter(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_retry = MagicMock(spec=IRetryHandler)
        self.mock_downloader = MagicMock(spec=IFileDownloader)
        self.mock_text_proc = MagicMock(spec=ITextProcessor)
        self.yt_path = "yt-dlp"

    @patch('subprocess.run')
    def test_get_safe_title_calls_subprocess(self, mock_run):
        """
        Verifikasi bahwa get_safe_title memanggil subprocess dengan parameter yang benar.
        """
        # Arrange
        with patch('pathlib.Path.exists', return_value=True):
            adapter = YouTubeAdapter(
                yt_dlp_path=self.yt_path,
                logger=self.mock_logger,
                cookies_path="dummy/cookies.txt"
            )
        test_url = "https://www.youtube.com/watch?v=test"
        
        # Mock successful execution
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Safe_Title"
        mock_run.return_value = mock_process

        # Act
        result = adapter.get_safe_title(test_url)

        # Assert
        self.assertEqual(result, "Safe_Title")
        mock_run.assert_called_once()
        
        cmd_args = mock_run.call_args[0][0]
        self.assertIn('--get-title', cmd_args)
        self.assertIn(test_url, cmd_args)
        self.assertIn('--cookies', cmd_args)

    @patch('subprocess.run')
    def test_get_safe_title_handles_subprocess_error(self, mock_run):
        """Verifikasi bahwa get_safe_title menangani subprocess.CalledProcessError."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Generic error")
        with self.assertRaises(MediaDownloadError):
            adapter.get_safe_title("http://url")
    
    @patch('subprocess.run')
    def test_get_safe_title_handles_timeout_error(self, mock_run):
        """Verifikasi bahwa get_safe_title menangani subprocess.TimeoutExpired."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
        with self.assertRaises(MediaDownloadError):
            adapter.get_safe_title("http://url")

    @patch('subprocess.run')
    def test_download_audio_calls_subprocess(self, mock_run):
        """Verifikasi download audio memanggil yt-dlp dengan argumen yang benar."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        mock_run.return_value.returncode = 0
        
        with patch.object(Path, 'mkdir'), \
             patch.object(Path, 'glob', return_value=[Path("out/test.wav")]) as mock_glob:
            
            result = adapter.download_audio("http://url", "out", "test")
            
            self.assertEqual(result, str(Path("out/test.wav")))
            args = mock_run.call_args[0][0]
            self.assertIn('-x', args)
            self.assertIn('--audio-format', args)

    @patch('subprocess.run')
    def test_download_audio_raises_error_if_output_missing(self, mock_run):
        """Verifikasi error jika file output tidak ditemukan setelah download."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        mock_run.return_value.returncode = 0
        
        with patch.object(Path, 'mkdir'), \
             patch.object(Path, 'glob', return_value=[]) as mock_glob: # Empty glob result
            
            with self.assertRaises(MediaDownloadError):
                adapter.download_audio("http://url", "out", "test")

    @patch('subprocess.run')
    def test_download_audio_handles_invalid_url(self, mock_run):
        """Verifikasi download audio menangani URL tidak valid."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        mock_run.side_effect = subprocess.CalledProcessError(1, "yt-dlp", stderr="Invalid URL")
        
        with patch.object(Path, 'mkdir'):
            with self.assertRaises(MediaDownloadError):
                adapter.download_audio("invalid_url", "out", "test")

    @patch('subprocess.run')
    def test_get_transcript_calls_subprocess(self, mock_run):
        """Verifikasi pengambilan transkrip."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        # Mock get_safe_title call inside get_transcript
        adapter.get_safe_title = MagicMock(return_value="Safe_Title")
        
        with patch.object(Path, 'mkdir'), \
             patch.object(Path, 'glob', return_value=[Path("out/Safe_Title.en.srt")]) as mock_glob:
             result = adapter.get_transcript("http://url", "out")
             self.assertEqual(result, str(Path("out/Safe_Title.en.srt")))

    @patch('subprocess.run')
    def test_get_transcript_raises_error_if_missing(self, mock_run):
        """Verifikasi error jika file transkrip tidak ditemukan."""
        adapter = YouTubeAdapter(self.yt_path, self.mock_logger)
        adapter.get_safe_title = MagicMock(return_value="Safe_Title")

        with patch.object(Path, 'mkdir'), \
             patch.object(Path, 'glob', return_value=[]) as mock_glob: # Empty glob result
             with self.assertRaises(MediaDownloadError):
                 adapter.get_transcript("http://url", "out")

class TestFFmpegAdapter(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)
        self.ffmpeg_path = "/usr/bin/ffmpeg"
        self.ffprobe_path = "/usr/bin/ffprobe"

    def test_encoder_priority_1_env_var(self):
        """
        Validasi Prioritas 1: Variabel Lingkungan FFMPEG_ENCODER harus diutamakan.
        """
        with patch.dict(os.environ, {"FFMPEG_ENCODER": "nvenc"}):
            adapter = FFmpegAdapter(
                ffmpeg_path=self.ffmpeg_path,
                ffprobe_path=self.ffprobe_path,
                logger=self.mock_logger,
                encoder_preference="qsv" # Preference config ini harus diabaikan
            )
            
            # Trigger initialization
            adapter.initialize()
            
            # Assert
            self.assertEqual(adapter._video_args, adapter.NVENC_ARGS)
            self.mock_logger.info.assert_any_call(
                "🚀 FFmpeg Adapter: Menggunakan encoder via Environment Variable (NVIDIA NVENC)."
            )

    def test_encoder_priority_2_config(self):
        """
        Validasi Prioritas 2: Config preference digunakan jika Env Var tidak ada.
        """
        with patch.dict(os.environ, {}, clear=True): # Pastikan env var bersih
            adapter = FFmpegAdapter(
                ffmpeg_path=self.ffmpeg_path,
                ffprobe_path=self.ffprobe_path,
                logger=self.mock_logger,
                encoder_preference="qsv"
            )
            
            adapter.initialize()
            
            # Assert
            self.assertEqual(adapter._video_args, adapter.QSV_ARGS)
            self.mock_logger.info.assert_any_call(
                "🚀 FFmpeg Adapter: Menggunakan encoder via Config (Intel QuickSync (QSV))."
            )

    def test_encoder_priority_3_auto_detect(self):
        """
        Validasi Prioritas 3: Deteksi otomatis dijalankan jika tidak ada Env Var atau Config.
        """
        with patch.dict(os.environ, {}, clear=True):
            adapter = FFmpegAdapter(
                ffmpeg_path=self.ffmpeg_path,
                ffprobe_path=self.ffprobe_path,
                logger=self.mock_logger,
                encoder_preference=None
            )
            
            # Mock _determine_best_encoder agar kita bisa memverifikasi pemanggilannya
            # dan menghindari eksekusi subprocess yang sebenarnya.
            with patch.object(adapter, '_determine_best_encoder') as mock_detect:
                mock_detect.return_value = ("Mock Encoder", ["-c:v", "mock_enc"])
                
                adapter.initialize()
                
                # Assert
                mock_detect.assert_called_once()
                self.assertEqual(adapter._video_args, ["-c:v", "mock_enc"])

    def test_determine_best_encoder_fallback_to_cpu(self):
        """
        Verifikasi fallback ke CPU jika semua hardware encoder gagal.
        """
        adapter = FFmpegAdapter(
            ffmpeg_path=self.ffmpeg_path,
            ffprobe_path=self.ffprobe_path,
            logger=self.mock_logger
        )

        # Mock _is_encoder_functional agar selalu mengembalikan False
        with patch.object(adapter, '_is_encoder_functional', return_value=False):
            adapter.initialize()
            
            # Assert
            self.assertEqual(adapter._video_args, adapter.CPU_VIDEO_ARGS)
            self.mock_logger.info.assert_any_call(
                "⚙️ FFmpeg Adapter: Tidak ada akselerasi hardware fungsional yang terdeteksi. Menggunakan CPU (libx264)."
            )

    @patch('subprocess.run')
    def test_run_command_handles_subprocess_error(self, mock_run):
        """Verifikasi bahwa _run_command menangani error subprocess dengan benar."""
        adapter = FFmpegAdapter(ffmpeg_path=self.ffmpeg_path, ffprobe_path=self.ffprobe_path, logger=self.mock_logger)
        
        # Simulasi return code non-zero (gagal)
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "FFmpeg Error Log"
        mock_run.return_value = mock_process

        with self.assertRaises(VideoProcessingError):
            adapter._run_command(["ffmpeg", "dummy"], "Test Command")

    @patch('subprocess.run')
    def test_run_command_handles_exception(self, mock_run):
        """Verifikasi bahwa _run_command menangani exception sistem."""
        adapter = FFmpegAdapter(ffmpeg_path=self.ffmpeg_path, ffprobe_path=self.ffprobe_path, logger=self.mock_logger)
        mock_run.side_effect = Exception("System Error")
        
        with self.assertRaises(VideoProcessingError):
            adapter._run_command(["ffmpeg", "dummy"], "Test Command")


class TestWhisperAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)

    @patch('src.infrastructure.adapters.whisper_adapter.WhisperModel')
    def test_transcribe_returns_segments(self, MockWhisperModel):
        """Verifikasi transcribe mengembalikan list TranscriptionSegment."""
        # Arrange
        mock_instance = MockWhisperModel.return_value
        
        # Mock result objects from faster-whisper
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello world"
        mock_segment.words = [
            MagicMock(word="Hello", start=0.0, end=0.5, probability=0.9),
            MagicMock(word="world", start=0.5, end=1.0, probability=0.9)
        ]
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)
        
        adapter = WhisperAdapter("small", "cpu", "int8", self.mock_logger)
        
        # Act
        result = adapter.transcribe("dummy.wav")
        
        # Assert
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TranscriptionSegment)
        self.assertEqual(result[0].text, "Hello world")
        self.assertEqual(len(result[0].words), 2)
        self.mock_logger.info.assert_called()

    @patch('src.infrastructure.adapters.whisper_adapter.WhisperModel')
    def test_transcribe_handles_error(self, MockWhisperModel):
        """Verifikasi bahwa error saat transkripsi memicu TranscriptionError."""
        # Arrange
        mock_instance = MockWhisperModel.return_value
        mock_instance.transcribe.side_effect = Exception("Whisper Failed")
        
        adapter = WhisperAdapter("small", "cpu", "int8", self.mock_logger)
        
        # Act & Assert
        with self.assertRaises(TranscriptionError):
            adapter.transcribe("dummy.wav")
        self.mock_logger.error.assert_called()

    @patch('src.infrastructure.adapters.whisper_adapter.torch')
    def test_detect_hardware_gpu_available(self, mock_torch):
        """Verifikasi deteksi hardware jika GPU tersedia."""
        # Arrange
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024**3) # 12GB VRAM
        mock_props.name = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        # Act
        config = WhisperAdapter.detect_hardware(self.mock_logger)
        
        # Assert
        self.assertEqual(config['model_size'], "large-v3")
        self.assertEqual(config['device'], "cuda")
        self.assertEqual(config['compute_type'], "float16")

    @patch('src.infrastructure.adapters.whisper_adapter.torch')
    def test_detect_hardware_gpu_low_vram(self, mock_torch):
        """Verifikasi fallback ke CPU jika VRAM GPU rendah."""
        # Arrange
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 2 * (1024**3) # 2GB VRAM
        mock_props.name = "Test Low GPU"
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        # Act
        config = WhisperAdapter.detect_hardware(self.mock_logger)
        
        # Assert
        self.assertEqual(config['model_size'], "small")
        self.assertEqual(config['device'], "cpu") # Seharusnya fallback ke CPU

    @patch('src.infrastructure.adapters.whisper_adapter.torch')
    def test_detect_hardware_no_gpu(self, mock_torch):
        """Verifikasi default ke CPU jika GPU tidak tersedia."""
        mock_torch.cuda.is_available.return_value = False
        
        config = WhisperAdapter.detect_hardware(self.mock_logger)
        
        self.assertEqual(config['device'], "cpu")
        self.assertEqual(config['model_size'], "small")

class TestAssSubtitleWriter(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)
        self.writer = AssSubtitleWriter(self.mock_logger)

    def test_format_timestamp(self):
        """Verifikasi format timestamp ASS (H:MM:SS.cc)."""
        self.assertEqual(self.writer._format_timestamp(61.5), "0:01:01.50")
        self.assertEqual(self.writer._format_timestamp(3665.0), "1:01:05.00")

    def test_write_karaoke_subtitles_writes_file(self):
        """Verifikasi bahwa file ditulis dengan header dan konten."""
        words = [TranscriptionWord(word="Test", start=0.0, end=1.0, probability=1.0)]
        segments = [TranscriptionSegment(start=0.0, end=1.0, text="Test", words=words)]
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.parent") as mock_parent: # Skip mkdir
                self.writer.write_karaoke_subtitles(segments, "output.ass", 5, 1920, 1080)
                
                mock_file.assert_called_with(Path("output.ass"), "w", encoding="utf-8")
                handle = mock_file()
                # Cek penulisan Header
                handle.write.assert_any_call(self.writer._generate_ass_header(1920, 1080))

class TestGeminiAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_text_proc = MagicMock(spec=ITextProcessor)
        self.mock_retry = MagicMock(spec=IRetryHandler)
        # Bypass retry logic
        self.mock_retry.execute.side_effect = lambda f, *args, **kwargs: f(*args, **kwargs)
        
    @patch('src.infrastructure.adapters.gemini_adapter.genai.Client')
    def test_analyze_content_flow(self, MockClient):
        """Verifikasi alur: Upload -> Generate -> Delete."""
        # Arrange
        mock_client_instance = MockClient.return_value
        adapter = GeminiAdapter("fake_key", ["model-a"], self.mock_text_proc, self.mock_retry, self.mock_logger)
        
        # Mock file upload & state check
        mock_file = MagicMock()
        mock_file.name = "files/123"
        mock_file.state = "ACTIVE"
        mock_file.uri = "gs://uri"
        mock_file.mime_type = "audio/wav"
        
        mock_client_instance.files.upload.return_value = mock_file
        mock_client_instance.files.get.return_value = mock_file
        
        # Mock generation response
        mock_response = MagicMock()
        mock_response.text = '{"video_title": "Test"}'
        mock_client_instance.models.generate_content.return_value = mock_response
        
        self.mock_text_proc.extract_json.return_value = {"video_title": "Test", "clips": []}
        
        with patch('pathlib.Path.exists', return_value=True):
            # Act
            result = adapter.analyze_content("trans", "audio.wav", "prompt")
            
            # Assert
            self.assertEqual(result.video_title, "Test")
            mock_client_instance.files.upload.assert_called()
            mock_client_instance.models.generate_content.assert_called()
            mock_client_instance.files.delete.assert_called_with(name="files/123")

    @patch('src.infrastructure.adapters.gemini_adapter.genai.Client')
    def test_analyze_content_handles_api_error(self, MockClient):
        """Verifikasi bahwa error API selama analisis ditangani dengan benar."""
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("API Error")

        adapter = GeminiAdapter("fake_key", ["model-a"], self.mock_text_proc, self.mock_retry, self.mock_logger)

        with self.assertRaises(AnalysisError):
            adapter.analyze_content("trans", "audio.wav", "prompt")
        self.mock_logger.error.assert_called()






class TestMediaPipeAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_retry = MagicMock(spec=IRetryHandler)

    @patch('src.infrastructure.adapters.mediapipe_adapter.vision')
    @patch('src.infrastructure.adapters.mediapipe_adapter.cv2')
    @patch('pathlib.Path.exists', return_value=True) # model check
    def test_track_and_crop_gpu_fallback(self, mock_exists, mock_cv2, mock_vision):
        """Verifikasi fallback ke CPU jika inisialisasi GPU gagal."""
        adapter = MediaPipeAdapter("model.task", self.mock_retry, self.mock_logger)
        
        # Mock VideoCapture
        mock_cap = mock_cv2.VideoCapture.return_value
        mock_cap.isOpened.return_value = True
        
        # Simulasi properti video. Karena cv2 di-mock, CAP_PROP_XXX juga berupa MagicMock.
        # Kita atur side_effect agar cap.get() mengembalikan nilai numerik,
        # bukan object mock (agar int() tidak error).
        mock_cap.get.side_effect = lambda prop: 100.0 
        
        # Mock read untuk mengembalikan False pada loop pertama agar tidak infinite loop
        mock_cap.read.return_value = (False, None)
        
        # Mock FaceLandmarker: Panggilan pertama raise error (GPU), kedua sukses (CPU)
        MockLandmarker = mock_vision.FaceLandmarker
        MockLandmarker.create_from_options.side_effect = [Exception("GPU Fail"), MagicMock()]
        
        with patch('pathlib.Path.mkdir'):
            adapter.track_and_crop("in.mp4", "out.mp4")
            
            # Assert create_from_options dipanggil 2 kali (sekali gagal, sekali sukses)
            self.assertEqual(MockLandmarker.create_from_options.call_count, 2)
            self.mock_logger.warning.assert_called() # Cek log warning fallback