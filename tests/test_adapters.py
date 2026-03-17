import unittest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path

from src.domain.exceptions import ExecutableNotFoundError
from src.infrastructure.adapters.youtube_adapter import YouTubeAdapter

class TestYouTubeAdapter(unittest.TestCase):

    @patch('src.infrastructure.common.utils.find_executable')
    def test_get_video_info_uses_correct_js_runtime_format(self, mock_find_executable):
        """
        Verifikasi bahwa adapter secara eksplisit menyetel js_runtimes dengan format
        yang benar untuk menghindari error 'runtime not found' dan 'invalid format'.
        """
        # Arrange
        # Mock path yang akan dikembalikan oleh find_executable
        mock_node_path = '/usr/bin/node'
        mock_find_executable.return_value = mock_node_path
        adapter = YouTubeAdapter(cookies_path="dummy/cookies.txt")
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_find_executable.assert_called_once_with("node")

        # Act
        with patch('yt_dlp.YoutubeDL') as MockYoutubeDL:
            # Atur mock agar tidak error saat dipanggil
            mock_instance = MockYoutubeDL.return_value
            mock_instance.__enter__.return_value.extract_info.return_value = {"title": "Test"}
            
            adapter.get_video_info(test_url)

            # Assert
            # Ambil argumen yang digunakan untuk menginisialisasi YoutubeDL
            called_opts = MockYoutubeDL.call_args.args[0]
            
            # Assert: Pastikan kita mengirim js_runtimes dengan format yang benar
            self.assertIn('js_runtimes', called_opts)
            self.assertIn('node', called_opts['js_runtimes'])
            mock_instance.__enter__.return_value.extract_info.assert_called_once_with(test_url, download=False)

    @patch('src.infrastructure.common.utils.find_executable')
    def test_init_gracefully_handles_no_node(self, mock_find_executable):
        """
        Verifikasi bahwa adapter tidak crash saat inisialisasi jika 'node' tidak ditemukan,
        dan opsi 'js_runtimes' tidak ditambahkan.
        """
        # Arrange
        mock_find_executable.side_effect = ExecutableNotFoundError("node not found")
        
        # Act & Assert
        # Pastikan tidak ada error yang muncul saat inisialisasi
        adapter = YouTubeAdapter(cookies_path="dummy/cookies.txt")
        mock_find_executable.assert_called_once_with("node")

        # Verifikasi bahwa opsi js_runtimes tidak ada
        with patch('yt_dlp.YoutubeDL') as MockYoutubeDL:
            mock_instance = MockYoutubeDL.return_value
            mock_instance.__enter__.return_value.extract_info.return_value = {"title": "Test"}
            adapter.get_video_info("some_url")
            called_opts = MockYoutubeDL.call_args.args[0]
            self.assertNotIn('js_runtimes', called_opts)