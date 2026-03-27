import unittest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

# Pastikan path aplikasi terbaca jika menjalankan manual
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.domain.exceptions import (
    AnalysisError, 
    AuthenticationError, 
    QuotaExceededError, 
    ContentPolicyViolationError
)

class TestGeminiAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.api_key = "dummy_api_key"
        self.model_names = ["gemini-1.5-flash", "gemini-pro"]
        
        # Patch library eksternal (google.genai)
        self.patcher_client = patch('src.infrastructure.adapters.gemini_adapter.genai.Client')
        self.patcher_types = patch('src.infrastructure.adapters.gemini_adapter.types')
        
        self.MockClient = self.patcher_client.start()
        self.MockTypes = self.patcher_types.start()
        
        # Setup Mock Enum untuk FileState
        self.MockTypes.FileState.ACTIVE = "ACTIVE"
        self.MockTypes.FileState.PROCESSING = "PROCESSING"
        self.MockTypes.FileState.FAILED = "FAILED"
        
        # Inisialisasi Adapter
        self.adapter = GeminiAdapter(self.api_key, self.model_names, self.mock_logger)

    def tearDown(self):
        self.patcher_client.stop()
        self.patcher_types.stop()

    def test_initialization(self):
        """Memastikan adapter diinisialisasi dengan konfigurasi yang benar."""
        self.assertEqual(self.adapter.api_key, "dummy_api_key")
        # Memastikan nama model dinormalisasi dengan prefix 'models/'
        self.assertEqual(self.adapter.model_names[0], "models/gemini-1.5-flash")
        self.MockClient.assert_called_once()

    def test_check_key_validity_success(self):
        """Test validasi key sukses."""
        mock_client_instance = MagicMock()
        self.MockClient.return_value = mock_client_instance
        
        # Simulasi list models mengembalikan iterator tidak kosong
        mock_client_instance.models.list.return_value = iter(["model1"])
        
        is_valid = GeminiAdapter.check_key_validity("valid_key")
        self.assertTrue(is_valid)

    def test_check_key_validity_failure(self):
        """Test validasi key gagal (exception dari Google)."""
        self.MockClient.side_effect = Exception("Invalid Key")
        
        is_valid = GeminiAdapter.check_key_validity("invalid_key")
        self.assertFalse(is_valid)

    @patch('src.infrastructure.adapters.gemini_adapter.Path')
    def test_upload_file_blocking_success(self, MockPath):
        """Test upload file dengan simulasi blocking loop (PROCESSING -> ACTIVE)."""
        mock_client = self.MockClient.return_value
        
        # Mock Path existence
        MockPath.return_value.exists.return_value = True
        MockPath.return_value.name = "audio.wav"

        # Mock respons upload awal
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/audio_wav"
        mock_client.files.upload.return_value = mock_uploaded_file

        # Mock polling status: 2x PROCESSING, lalu 1x ACTIVE
        state_processing = MagicMock(state="PROCESSING")
        state_active = MagicMock(state="ACTIVE", name="files/audio_wav")
        
        # Side effect mensimulasikan pemanggilan berulang di dalam while loop
        mock_client.files.get.side_effect = [state_processing, state_processing, state_active]

        result = self.adapter.upload_file("dummy_path.wav")

        self.assertEqual(result, state_active)
        self.assertEqual(mock_client.files.get.call_count, 3) # Loop berjalan 3 kali
        self.mock_logger.info.assert_called() # Pastikan logging info dipanggil

    @patch('src.infrastructure.adapters.gemini_adapter.Path')
    def test_upload_file_failed_state(self, MockPath):
        """Test jika file berubah status menjadi FAILED."""
        mock_client = self.MockClient.return_value
        MockPath.return_value.exists.return_value = True

        mock_client.files.upload.return_value = MagicMock(name="files/fail")
        
        # Langsung return status FAILED
        state_failed = MagicMock(state="FAILED")
        state_failed.error.message = "Corrupt video"
        mock_client.files.get.return_value = state_failed

        with self.assertRaises(AnalysisError) as context:
            self.adapter.upload_file("fail.mp4")
        
        self.assertIn("File processing failed", str(context.exception))

    def test_generate_content_success(self):
        """Test generate content standar."""
        mock_client = self.MockClient.return_value
        mock_response = MagicMock()
        mock_response.text = "AI Result"
        mock_client.models.generate_content.return_value = mock_response

        result = self.adapter.generate_content("prompt")
        self.assertEqual(result, "AI Result")

    def test_generate_content_with_schema(self):
        """Test generate content dengan Structured Output (JSON Schema)."""
        mock_client = self.MockClient.return_value
        mock_response = MagicMock()
        # Simulasi response.parsed yang diisi otomatis oleh SDK jika schema diberikan
        expected_json = {"summary": "test"}
        mock_response.parsed = expected_json
        mock_client.models.generate_content.return_value = mock_response

        dummy_schema = MagicMock()
        result = self.adapter.generate_content("prompt", response_schema=dummy_schema)
        
        self.assertEqual(result, expected_json)
        # Verifikasi config schema dikirim
        _, kwargs = mock_client.models.generate_content.call_args
        self.assertIsNotNone(kwargs['config'])

    def test_error_mapping(self):
        """Test pemetaan Exception Google ke Domain Exception."""
        # 1. Auth Error
        with self.assertRaises(AuthenticationError):
            self.adapter._handle_gemini_error(Exception("403 Forbidden"), "test")

        # 2. Quota Error
        with self.assertRaises(QuotaExceededError):
            self.adapter._handle_gemini_error(Exception("429 Resource exhausted"), "test")

        # 3. Safety Error
        with self.assertRaises(ContentPolicyViolationError):
            self.adapter._handle_gemini_error(Exception("Content blocked by safety filters"), "test")

        # 4. General Error
        with self.assertRaises(AnalysisError):
            self.adapter._handle_gemini_error(Exception("Unknown error"), "test")

    def test_delete_file(self):
        """Test penghapusan file."""
        mock_client = self.MockClient.return_value
        self.adapter.delete_file("files/123")
        mock_client.files.delete.assert_called_with(name="files/123")

if __name__ == '__main__':
    unittest.main()