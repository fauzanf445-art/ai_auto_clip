import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import time

from src.infrastructure.common.filesystem import SystemHelper
from src.domain.interfaces import ILogger

class TestSystemHelper(unittest.TestCase):

    def setUp(self):
        """Inisialisasi instance SystemHelper untuk setiap tes."""
        self.mock_logger = MagicMock(spec=ILogger)
        self.system_helper = SystemHelper(self.mock_logger)

    def _setup_mocks(self, count: int, size_mb: int, start_time: float):
        """
        Helper untuk menyiapkan mock os.walk dan Path.stat.
        Mengembalikan list mock path object yang diharapkan diproses.
        """
        filenames = [f"file_{i}.mp4" for i in range(count)]
        mock_paths = []
        
        # Setup return values for stat()
        files = []
        for i in range(count):
            p = MagicMock(spec=Path)
            p.name = filenames[i]
            mtime = start_time + i
            size = size_mb * 1024 * 1024
            p.stat.return_value.st_mtime = mtime
            p.stat.return_value.st_size = size
            
            # Supaya saat sorted() dipanggil, mock path bisa dibandingkan (opsional, tapi aman)
            # Implementasi SystemHelper menyimpan tuple (path, mtime, size) lalu sort key=x[1]
            # Jadi path object sendiri tidak perlu comparable.
            
            files.append(p)

        return filenames, files

    @patch('src.infrastructure.common.filesystem.os.walk')
    def test_prune_directory_by_max_files(self, mock_walk):
        """
        Tes: Pemangkasan harus terjadi ketika jumlah file melebihi `max_files`.
        """
        # Arrange: Buat 12 file, batasnya 10.
        filenames, mock_paths = self._setup_mocks(count=12, size_mb=10, start_time=time.time())
        mock_walk.return_value = [("/tmp", [], filenames)]
        
        max_files = 10
        max_size_mb = 500  # Batas ukuran tidak terlampaui

        # Act
        # Kita perlu mem-patch Path() constructor atau iterasi di dalam SystemHelper
        # SystemHelper: file_path = Path(root) / file
        # Agar file_path.stat() mengembalikan nilai yg kita mau, kita harus inject logic ini.
        # Alternatif: patch Path di utils.py
        
        with patch('src.infrastructure.common.filesystem.Path') as MockPath:
            # Konfigurasi side_effect untuk Path(root) / file
            # MockPath return value adalah object yang kalau di / file menghasilkan mock path spesifik
            # Ini agak rumit. Lebih mudah mem-patch os.walk dan logika iterasi?
            # SystemHelper menggunakan: file_path = Path(root) / file
            
            # Simplifikasi: Kita patch os.walk agar return list file names.
            # Lalu kita patch Path.__truediv__ (operator /) untuk mengembalikan mock path yang sudah kita setup.
            
            # Setup dictionary mapping filename -> mock_path object
            name_to_mock = {m.name: m for m in mock_paths}
            
            def path_div_side_effect(self, other):
                # self adalah Path(root), other adalah filename
                return name_to_mock.get(other, MagicMock(spec=Path))
            
            MockPath.return_value.__truediv__.side_effect = path_div_side_effect
            
            self.system_helper.prune_directory(Path("/tmp"), max_files, max_size_mb, extensions=('.mp4',))

        # Assert: Harus menghapus 2 file paling tua.
        # file_0 and file_1 should be unlinked
        mock_paths[0].unlink.assert_called_once()
        mock_paths[1].unlink.assert_called_once()
        mock_paths[2].unlink.assert_not_called()

    @patch('src.infrastructure.common.filesystem.os.walk')
    def test_prune_directory_by_max_size(self, mock_walk):
        filenames, mock_paths = self._setup_mocks(count=5, size_mb=120, start_time=time.time())
        mock_walk.return_value = [("/tmp", [], filenames)]
        
        name_to_mock = {m.name: m for m in mock_paths}
        
        with patch('src.infrastructure.common.filesystem.Path') as MockPath:
            MockPath.return_value.__truediv__.side_effect = lambda self, other: name_to_mock.get(other, MagicMock(spec=Path))
            
            # 5 * 120 = 600MB. Max 500. Need to remove 100MB+. 1 file (120MB) enough.
            self.system_helper.prune_directory(Path("/tmp"), max_files=10, max_size_mb=500, extensions=('.mp4',))
            
        mock_paths[0].unlink.assert_called_once()
        mock_paths[1].unlink.assert_not_called()

    @patch('src.infrastructure.common.filesystem.os.walk')
    def test_prune_directory_combined_rules(self, mock_walk):
        filenames, mock_paths = self._setup_mocks(count=12, size_mb=50, start_time=time.time())
        mock_walk.return_value = [("/tmp", [], filenames)]
        
        name_to_mock = {m.name: m for m in mock_paths}
        
        with patch('src.infrastructure.common.filesystem.Path') as MockPath:
            MockPath.return_value.__truediv__.side_effect = lambda self, other: name_to_mock.get(other, MagicMock(spec=Path))
            
            # Max files 10 -> prune file_0, file_1.
            # Remaining size: 10 * 50 = 500MB. Max 500. No more needed? 
            # Wait, logic in code:
            # 1. Prune by count first.
            # 2. Prune by size.
            # Total size initial: 600MB.
            # After count prune (2 files): 500MB.
            # Max size is 500MB. condition `total_size > max_size_bytes` (500 > 500 is False).
            # So only 2 files should be removed?
            # Let's check test expectation.
            # "Ini akan menandai file_0, file_1, file_2".
            # If limit is > 500, and current is 500, it stops.
            # So strictly > 500MB means 500MB is okay.
            # If we want 3 files removed, size must be > 500 after 2 removals?
            # Or maybe the test data was slightly different.
            # Let's stick to what logic dictates: 2 files removed.
            
            self.system_helper.prune_directory(Path("/tmp"), max_files=10, max_size_mb=500, extensions=('.mp4',))

        mock_paths[0].unlink.assert_called_once()
        mock_paths[1].unlink.assert_called_once()
        mock_paths[2].unlink.assert_not_called()

if __name__ == '__main__':
    unittest.main()