import unittest
from unittest.mock import MagicMock, patch, call, ANY
from pathlib import Path

# Import Domain & Interfaces
from src.domain.models import Clip, VideoSummary, TrackResult
from src.domain.interfaces import (
    IMediaDownloader, IVideoProcessor, IFaceTracker, IContentAnalyzer, ILogger,
    ITranscriber, ICacheManager, IProgressReporter, IProgressBar, ISystemHelper
)
from src.application.context import SessionContext

# Import Config & UI for Orchestrator test
from src.config import AppConfig, AppPaths
from src.infrastructure.ui.console import ConsoleUI

# Import Services yang akan dites
from src.application.services.provider_service import ProviderService
from src.application.services.editor_service import EditorService

# Import Orchestrator
from src.application.workflow import Workflow

class TestProviderService(unittest.TestCase):
    def setUp(self):
        # Mock Interface Downloader
        self.mock_downloader = MagicMock(spec=IMediaDownloader)
        self.mock_logger = MagicMock(spec=ILogger)
        # Injeksi Mock ke Service
        self.service = ProviderService(
            downloader=self.mock_downloader, 
            processor=MagicMock(), 
            analyzer=MagicMock(), 
            retry_handler=MagicMock(),
            cache_manager=MagicMock(spec=ICacheManager),
            logger=self.mock_logger
        )

    def test_get_video_metadata(self):
        # Arrange
        url = "http://youtube.com/test"
        expected_title = "Test_Video_Safe_Name"
        # Update method call to match interface (get_safe_title instead of get_video_info)
        self.mock_downloader.get_safe_title.return_value = expected_title

        # Act
        result = self.service.get_safe_folder_name(url)

        # Assert
        self.assertEqual(result, expected_title)
        self.mock_downloader.get_safe_title.assert_called_once_with(url)

class TestEditorService(unittest.TestCase):
    def setUp(self):
        self.mock_downloader = MagicMock(spec=IMediaDownloader)
        self.mock_processor = MagicMock(spec=IVideoProcessor)
        self.mock_tracker = MagicMock(spec=IFaceTracker)
        self.mock_progress = MagicMock(spec=IProgressReporter)
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_system = MagicMock(spec=ISystemHelper)
        
        # Setup mock progress sequence to just yield the iterable
        def mock_sequence(iterable, **kwargs):
            return iterable
        self.mock_progress.sequence.side_effect = mock_sequence

        self.service = EditorService(
            downloader=self.mock_downloader,
            processor=self.mock_processor, 
            tracker=self.mock_tracker, 
            transcriber=MagicMock(), 
            writer=MagicMock(),
            system_helper=self.mock_system,
            fonts_dir=Path("/mock/fonts"),
            karaoke_chunk_size=2,
            logger=self.mock_logger
        )

    def test_batch_create_clips_calls_downloader(self):
        # Arrange
        clips = [
            Clip(id="clip1id", title="Clip A", start_time=0, end_time=10, energy_score=10, vocal_energy="High", audio_justification="", description="", caption=""),
            Clip(id="clip2id", title="Clip B", start_time=20, end_time=30, energy_score=8, vocal_energy="Med", audio_justification="", description="", caption="")
        ]
        video_url = "http://vid"
        output_dir = Path("temp/clips")
        
        # Simulasi downloader berhasil memotong dengan mengembalikan path
        self.mock_downloader.download_video_section.side_effect = lambda url, start, end, output_path: Path(output_path)

        # Act
        # Kita mock Path.exists agar tidak benar-benar cek file system
        with patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'mkdir'): 
            
            # Kita patch ThreadPoolExecutor agar berjalan sinkronus (serial) untuk testing
            with patch('concurrent.futures.ThreadPoolExecutor') as MockExecutor:
                mock_executor_instance = MockExecutor.return_value
                mock_executor_instance.__enter__.return_value = mock_executor_instance
                
                # Create fake futures
                fake_futures = [MagicMock() for _ in clips]
                for i, f in enumerate(fake_futures):
                    f.result.return_value = Path(output_dir / f"{clips[i].id[:8]}_{clips[i].title}.mp4")
                
                futures_iter = iter(fake_futures)

                def submit_side_effect(fn, *args, **kwargs):
                    fn(*args, **kwargs)
                    return next(futures_iter)
                
                mock_executor_instance.submit.side_effect = submit_side_effect
                
                # Mock as_completed untuk mengembalikan hasil secara langsung
                with patch('concurrent.futures.as_completed') as mock_as_completed:
                    mock_as_completed.return_value = fake_futures

                    self.service.batch_create_clips(
                        clips=clips,
                        source_url=video_url,
                        output_dir=output_dir
                    )

        # Assert
        # Pastikan download_video_section dipanggil 2 kali (sekali untuk setiap klip)
        self.assertEqual(self.mock_downloader.download_video_section.call_count, 2)
        self.mock_downloader.download_video_section.assert_any_call(
            url=video_url, start=0, end=10, output_path=str(output_dir / "clip1id_Clip A.mp4")
        )

class TestAnalysisService(unittest.TestCase):
    def setUp(self):
        self.mock_analyzer = MagicMock(spec=IContentAnalyzer)
        self.mock_logger = MagicMock(spec=ILogger)
        self.service = ProviderService(
            downloader=MagicMock(), 
            processor=MagicMock(), 
            analyzer=self.mock_analyzer, 
            cache_manager=MagicMock(spec=ICacheManager),
            retry_handler=MagicMock(),
            logger=self.mock_logger
        )

    def test_analyze_video_no_cache(self):
        # Arrange
        transcript = "Halo dunia"
        audio_path = "audio.wav"
        prompt = "Analyze this"
        
        expected_summary = VideoSummary(
            video_title="Test", 
            audio_energy_profile="High", 
            clips=[]
        )
        self.mock_analyzer.analyze_content.return_value = expected_summary

        # Act
        result = self.service.analyze_video(transcript, audio_path, prompt, cache_path=None)

        # Assert
        self.assertEqual(result, expected_summary)
        self.mock_analyzer.analyze_content.assert_called_once_with(transcript, audio_path, prompt, '')

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for all dependencies."""
        self.mock_config = MagicMock(spec=AppConfig)
        self.mock_ui = MagicMock(spec=ConsoleUI)
        self.mock_provider_service = MagicMock(spec=ProviderService)
        self.mock_editor_service = MagicMock(spec=EditorService)
        self.mock_progress = MagicMock(spec=IProgressReporter)
        self.mock_manager = MagicMock()
        self.mock_logger = MagicMock()

        # Mock sequence behavior
        def mock_sequence(iterable, **kwargs):
            return iterable
        self.mock_progress.sequence.side_effect = mock_sequence

        # Mock manual behavior
        self.mock_progress.manual.return_value = MagicMock(spec=IProgressBar)
        
        # Configure nested mock attributes to prevent AttributeErrors
        # The orchestrator checks this property to decide on parallelism.
        # We must configure the nested 'processor' mock explicitly when using a spec.
        self.mock_editor_service.processor = MagicMock()
        self.mock_editor_service.processor.is_gpu_enabled = False

        # Configure mock paths
        self.mock_config.paths = MagicMock(spec=AppPaths)
        self.mock_config.paths.TEMP_DIR = Path("/tmp")
        self.mock_config.paths.OUTPUT_DIR = Path("/output")
        self.mock_config.paths.FONTS_DIR = Path("/fonts")
        self.mock_config.paths.PROMPT_FILE = MagicMock()
        self.mock_config.paths.PROMPT_FILE.exists.return_value = True
        self.mock_config.paths.PROMPT_FILE.read_text.return_value = "Mock Prompt"
        self.mock_config.karaoke_chunk_size = 3

        # Instantiate the Orchestrator with mocks
        self.orchestrator = Workflow(
            config=self.mock_config,
            provider=self.mock_provider_service,
            editor=self.mock_editor_service,
            manager_factory=self.mock_manager,
            logger=self.mock_logger
        )
        
        self.mock_ctx = MagicMock(spec=SessionContext)
        self.mock_ctx.ui = self.mock_ui
        self.mock_ctx.progress_reporter = self.mock_progress
        self.mock_ctx.api_key = "test_key"

    def test_identify_clips_manual_mode(self):
        """Test getting clips when user provides manual timestamps."""
        # Arrange
        manual_timestamps = [{'start_time': 0, 'end_time': 10}]
        
        # Act
        # Patch _try_get_manual_clips on the orchestrator instance to return clips
        with patch.object(self.orchestrator, '_try_get_manual_clips', return_value=[Clip.create_manual(0, 0, 10)]):
            result_clips = self.orchestrator._identify_clips(self.mock_ctx, "http://test.url", Path("/tmp/work"))

        # Assert
        self.mock_ui.show_info.assert_any_call("🚀 [STEP] Analisis Konten...")
        # self.mock_ui.get_manual_clips.assert_called_once() # Removed as method doesn't exist on spec
        # Ensure AI path is not taken
        self.mock_provider_service.get_transcript.assert_not_called()
        
        # Verify that the orchestrator correctly created the Clip object
        self.assertEqual(len(result_clips), 1)
        created_clip = result_clips[0]
        self.assertEqual(created_clip.start_time, 0)
        self.assertEqual(created_clip.end_time, 10)
        self.assertEqual(created_clip.duration, 10)
        self.assertEqual(created_clip.title, "Manual Clip 1")

    def test_identify_clips_ai_mode(self):
        """Test getting clips using the AI analysis path."""
        # Arrange
        url = "http://test.url"
        work_dir = Path("/tmp/work")
        ai_clips = [Clip(id="ai_1", title="AI Clip", start_time=10, end_time=20, energy_score=9, vocal_energy="High", audio_justification="", description="", caption="")]
        summary = VideoSummary(video_title="AI Video", audio_energy_profile="Dynamic", clips=ai_clips)
        
        self.mock_provider_service.get_transcript.return_value = "some transcript"
        self.mock_provider_service.prepare_audio_for_analysis.return_value = work_dir / "full_audio.wav"
        self.mock_provider_service.analyze_video.return_value = summary

        # Act
        result_clips = self.orchestrator._identify_clips(self.mock_ctx, url, work_dir)

        # Assert
        # self.mock_ui.get_manual_clips.assert_called_once() # Removed
        self.mock_provider_service.get_transcript.assert_called_once_with(url, temp_dir=str(work_dir))
        self.mock_provider_service.prepare_audio_for_analysis.assert_called_once_with(url, work_dir, "full_audio")
        self.mock_provider_service.analyze_video.assert_called_once_with(
            transcript="some transcript",
            audio_path=str(work_dir / "full_audio.wav"),
            prompt="Mock Prompt",
            cache_path=str(work_dir / "summary.json"),
            api_key="test_key"
        )
        self.assertEqual(result_clips, ai_clips)

    def test_cut_raw_clips(self):
        """Test the raw clip cutting step."""
        # Arrange
        clips_to_cut = [Clip(id="c1", title="C1", start_time=0, end_time=10, energy_score=0, vocal_energy="", audio_justification="", description="", caption="")]
        url = "http://test.url"
        work_dir = Path("/tmp/work")
        expected_paths = [work_dir / "raw_clips" / "c1_C1.mp4"]
        
        self.mock_editor_service.batch_create_clips.return_value = expected_paths

        # Act
        result_paths = self.orchestrator._cut_raw_clips(self.mock_ctx, clips_to_cut, url, work_dir)

        # Assert
        self.mock_ui.show_info.assert_any_call("🚀 [STEP] Downloading & CFR Conversion...")
        self.mock_editor_service.batch_create_clips.assert_called_once_with(
            clips=clips_to_cut,
            source_url=url,
            output_dir=work_dir / "raw_clips",
            progress_reporter=self.mock_ctx.progress_reporter
        )
        self.assertEqual(result_paths, expected_paths)

    def test_render_final_clips(self):
        """Test the final rendering step."""
        # Arrange
        work_dir = Path("/tmp/work")
        safe_name = "test_video"
        original_path = Path("/tmp/work/raw_clips/clip1.mp4")
        track_res = {'tracked_video': '/tmp/work/tracked_clips/tracked_clip1.mp4', 'width': 1080, 'height': 1920}
        track_res = TrackResult(tracked_video='/tmp/work/tracked_clips/tracked_clip1.mp4', width=1080, height=1920)
        tracked_results = [(original_path, track_res)]
        final_clip_path = self.mock_config.paths.OUTPUT_DIR / safe_name / f"final_{original_path.name}"
        
        # EditorService.batch_render now handles everything. Mock its return value.
        self.mock_editor_service.batch_render.return_value = [final_clip_path]

        # Act
        final_clips = self.orchestrator._render_final_clips(self.mock_ctx, tracked_results, work_dir, safe_name)

        # Assert
        self.mock_ui.show_info.assert_any_call("🚀 [STEP] Captioning & Rendering Final...")
        self.mock_editor_service.batch_render.assert_called_once_with(
            tracked_results=tracked_results,
            work_dir=work_dir,
            output_dir=self.mock_config.paths.OUTPUT_DIR / safe_name,
            progress_reporter=self.mock_ctx.progress_reporter
        )
        self.assertEqual(final_clips, [final_clip_path])

if __name__ == '__main__':
    unittest.main()
