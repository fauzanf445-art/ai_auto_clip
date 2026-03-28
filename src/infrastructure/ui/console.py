import logging
import getpass
from typing import Any, Callable

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

    def show_info(self, msg: str, level: str = "INFO"):
        """Menampilkan pesan informasi (bisa step, log, atau success)."""
        logging.info(msg)

    def show_error(self, msg: str):
        """Menampilkan pesan error."""
        logging.error(f"❌ ERROR: {msg}")

    @property
    def log_output(self) -> str:
        """Console UI tidak mengumpulkan output log dalam satu string buffer."""
        return ""

    def log(self, msg: str): self.show_info(f"   -> {msg}")
    
    def show_step(self, msg: str): self.show_info(f"🚀 [STEP] {msg}...")

    def create_demo(self, process_fn: Callable, cleanup_fn: Callable, default_api_key: str = "") -> Any:
        """Mode CLI tidak mendukung pembuatan Gradio Demo."""
        return None
