import logging
import getpass

from src.domain.interfaces import IUserInterface

class ConsoleUI(IUserInterface):
    """Antarmuka Pengguna berbasis Terminal."""

    def print_banner(self):
        print("\n" + "="*40)
        print("   🎬 HSU AI CLIPPER - CLEAN ARCH   ")
        print("="*40 + "\n")

    def get_input(self, prompt: str) -> str:
        """Mengambil input teks standar dari user."""
        return input(f"\n👉 {prompt}: ").strip()

    def get_secure_input(self, prompt: str) -> str:
        """Mengambil input sensitif (password/key) tanpa echo."""
        return getpass.getpass(f"👉 {prompt}: ").strip()

    def show_info(self, msg: str):
        """Menampilkan pesan informasi (bisa step, log, atau success)."""
        logging.info(msg)

    def show_error(self, msg: str):
        """Menampilkan pesan error."""
        logging.error(f"❌ ERROR: {msg}")

    # Compatibility Methods (Bisa dihapus nanti setelah refactor Workflow selesai)
    # Agar tidak mematahkan kode lama seketika
    def log(self, msg: str): self.show_info(f"   -> {msg}")
    def show_step(self, msg: str): self.show_info(f"🚀 [STEP] {msg}...")
