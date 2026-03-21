import unittest
from unittest.mock import patch
import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# Config & UI
from src.config import AppConfig
from src.infrastructure.ui.console import ConsoleUI
from src.bootstrap import Bootstrap
from src.container import Container
from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.application.context import SessionContext
from src.infrastructure.ui.progress import LogProgressReporter
from src.domain.models import Clip

# --- Konfigurasi Tes ---
# Gunakan video pendek dan publik untuk konsistensi
TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # "Me at the zoo"
TEST_VIDEO_SAFE_NAME = "Me_at_the_zoo" # Sesuai sanitasi YouTubeAdapter (spasi -> underscore)

# Muat API key dari .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

@unittest.skipIf(not API_KEY, "GEMINI_API_KEY tidak ditemukan di .env. Tes integrasi dilewati.")
class TestSystemIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Mempersiapkan semua instance nyata yang diperlukan untuk pipeline.
        Ini seperti mini `main.py`.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        cls.config = AppConfig()

        # Replikasi setup environment yang sekarang ada di main.py
        # untuk memastikan test berjalan dalam kondisi yang sama.
        Bootstrap.setup_directories(cls.config)

        cls.ui = ConsoleUI() # Kita tetap butuh instance-nya, meski tidak akan menampilkan ke user

        # Hapus output lama sebelum memulai
        shutil.rmtree(cls.config.paths.TEMP_DIR / TEST_VIDEO_SAFE_NAME, ignore_errors=True)
        shutil.rmtree(cls.config.paths.OUTPUT_DIR / TEST_VIDEO_SAFE_NAME, ignore_errors=True)

        # Inisialisasi via Container
        cls.container = Container(cls.config)
        
        # Pastikan cookies di-setup dari environment variable (Secret)
        cls.container.auth_service.check_and_setup_cookies(cls.config.paths.COOKIE_FILE)
        cls.orchestrator = cls.container.orchestrator
        cls.provider = cls.container.provider_service

    @classmethod
    def tearDownClass(cls):
        """Membersihkan file yang dihasilkan setelah tes selesai."""
        print("\nMembersihkan file tes...")
        temp_dir = cls.config.paths.TEMP_DIR / TEST_VIDEO_SAFE_NAME
        output_dir = cls.config.paths.OUTPUT_DIR / TEST_VIDEO_SAFE_NAME
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Dihapus: {temp_dir}")
        print(f"Dihapus: {output_dir}")

    def test_1_gemini_integration(self):
        """Verifikasi bahwa Gemini API Key valid dan service dapat dihubungi."""
        is_valid = GeminiAdapter.check_key_validity(API_KEY or "")
        self.assertTrue(is_valid, "Gemini API Key validasi gagal (cek koneksi/kuota).")

    def test_2_youtube_integration(self):
        """Verifikasi kemampuan mengambil metadata dari YouTube."""
        # Tes ini memastikan yt-dlp bekerja dan internet terhubung
        safe_title = self.provider.get_safe_folder_name(TEST_URL)
        self.assertIsNotNone(safe_title)
        # Verifikasi sanitasi nama folder (pastikan spasi diganti underscore)
        self.assertEqual(safe_title, TEST_VIDEO_SAFE_NAME)

    def test_3_run_pipeline_end_to_end_with_manual_clip(self):
        """
        Menjalankan pipeline lengkap dari awal hingga akhir menggunakan mode manual
        untuk mempercepat proses (menghindari panggilan AI yang mahal dan lama).
        """
        # Arrange
        # Kita "memalsukan" input pengguna dengan mem-patch metode di orchestrator
        # agar langsung mengembalikan objek Clip tanpa interaksi UI.
        manual_clip = Clip.create_manual(0, 2.0, 5.0)

        with patch.object(self.orchestrator, '_try_get_manual_clips', return_value=[manual_clip]):

            # Create context for the run
            ctx = SessionContext(
                ui=self.ui,
                api_key=API_KEY or "",
                progress_reporter=LogProgressReporter(self.container.logger)
            )

            # Act
            self.orchestrator.run(ctx, TEST_URL)

        # Assert
        # Verifikasi bahwa file output benar-benar dibuat
        output_dir = self.config.paths.OUTPUT_DIR / TEST_VIDEO_SAFE_NAME
        self.assertTrue(output_dir.exists(), "Folder output utama seharusnya dibuat.")

        final_clips = list(output_dir.glob("final_*.mp4"))
        self.assertGreater(len(final_clips), 0, "Seharusnya ada setidaknya satu file video final yang dirender.")
        
        first_clip = final_clips[0]
        self.assertGreater(first_clip.stat().st_size, 10 * 1024, f"File {first_clip.name} terlihat terlalu kecil (kurang dari 10KB).")
        print(f"\n✅ Verifikasi berhasil: File output '{first_clip.name}' dibuat dan valid.")

if __name__ == '__main__':
    unittest.main()
