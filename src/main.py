import os
import argparse
from typing import Optional
from dotenv import load_dotenv

from src.config import AppConfig
from src.infrastructure.ui.console import ConsoleUI
from src.infrastructure.ui.gradio_ui import create_gradio_interface, GradioUI
from src.infrastructure.ui.progress import LogProgressReporter, TqdmProgressReporter
from src.infrastructure.ui.logging_config import TqdmLogger
from src.container import Container
from src.application.context import SessionContext

# Suppress TensorFlow/MediaPipe C++ logging to keep UI clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Mengembalikan instance Singleton dari AppConfig."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def setup_environment(config: AppConfig):
    """Mempersiapkan lingkungan eksekusi."""
    for directory in config.paths.all_directories:
        directory.mkdir(parents=True, exist_ok=True)

def run_cli(url: Optional[str] = None, keep_temp: bool = False):
    """Menjalankan aplikasi dalam mode CLI."""
    config = get_config()
    setup_environment(config)
    
    ui = ConsoleUI()
    ui.print_banner()

    load_dotenv(config.paths.env_file)
    # Tidak lagi memaksa API Key di awal. Workflow yang akan menanganinya jika kosong.
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        ui.show_info("✅ API Key dimuat dari environment.")

    try:
        logger = TqdmLogger(config.paths.log_file, verbose=False)
        container = Container(config, logger, keep_temp=keep_temp)
        
        # Jalankan pengecekan aset (Download model/font jika belum ada)
        container.manager_service.ensure_system_integrity()
        
        ctx = SessionContext(
            ui=ui,
            api_key=api_key,
            progress_reporter=TqdmProgressReporter()
        )
        
        if not url:
            url = ui.get_input("Masukkan URL Video YouTube")
        
        url = url.split('#')[0].strip()
        if not url:
            ui.show_error("URL menjadi kosong setelah sanitasi. Harap berikan URL video yang valid.")
            return 
            
        container.workflow.run(ctx, url)

    except KeyboardInterrupt:
        print("\n👋 Dibatalkan pengguna.")

def run_web():
    """Menjalankan aplikasi dalam mode Web dengan dukungan penuh Google Colab."""
    print("🌐 Memulai HSU AI Clipper dalam mode Web (Gradio)... [cite: 4]")
    config = get_config() [cite: 4]
    setup_environment(config) [cite: 4]

    # --- Logika Deteksi Environment ---
    is_huggingface = os.getenv("SPACE_ID") is not None [cite: 4]
    
    # Deteksi apakah berjalan di Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    def process_via_web(url, api_key, keep_temp): [cite: 4]
        if not url: [cite: 4]
            yield "Error: URL YouTube wajib diisi.", None [cite: 4]
            return [cite: 4]
        if not api_key: [cite: 4]
            yield "Error: Gemini API Key wajib diisi.", None [cite: 4]
            return [cite: 4]

        ui = GradioUI() [cite: 4]
        ui.log("🚀 Memulai inisialisasi sistem...") [cite: 4]
        yield ui.log_output, None [cite: 4]

        try:
            # Logger dan Progress Reporter [cite: 4]
            logger = TqdmLogger(config.paths.log_file) [cite: 4]
            progress_reporter = LogProgressReporter(logger) [cite: 4]

            container = Container(config, logger, keep_temp=keep_temp) [cite: 4]
            
            # Memastikan aset (model/font) tersedia [cite: 4]
            container.manager_service.ensure_system_integrity() [cite: 4]
            
            ctx = SessionContext( [cite: 4]
                ui=ui,
                api_key=api_key,
                progress_reporter=progress_reporter
            ) [cite: 4]
            
            url = url.split('#')[0].strip() [cite: 4]
            ui.log(f"Memproses URL: {url}") [cite: 4]
            yield ui.log_output, None [cite: 4]
            
            video_files = container.workflow.run(ctx, url) [cite: 4]
            
            ui.log("✅ Proses Selesai!") [cite: 4]
            yield ui.log_output, [str(v) for v in video_files] [cite: 4]
            
        except Exception as e: [cite: 4]
            ui.show_error(str(e)) [cite: 4]
            yield ui.log_output, None [cite: 4]

    # Ambil API Key dari environment jika ada [cite: 4]
    default_api_key = os.getenv("GEMINI_API_KEY", "") [cite: 4]
    
    # Inisialisasi Interface [cite: 4]
    demo = create_gradio_interface(process_fn=process_via_web, default_api_key=default_api_key) [cite: 4]
    
    # --- Konfigurasi Network untuk Colab vs Lokal/Docker ---
    # 1. server_name: "0.0.0.0" agar bisa diakses dari luar container [cite: 2, 4]
    # 2. share: True WAJIB untuk Google Colab agar muncul public link .gradio.live
    server_name = "0.0.0.0" if (is_huggingface or is_colab) else "127.0.0.1" [cite: 4]
    should_share = True if is_colab else False

    print(f"🔧 Config: Colab={is_colab}, Share={should_share}, Host={server_name}")
    
    demo.queue().launch(
        server_name=server_name, 
        server_port=7860, [cite: 2, 4]
        share=should_share,
        debug=is_colab # Membantu melihat error detail di console Colab
    )

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="HSU AI Clipper - Automated Video Shorts Generator")
    parser.add_argument("url", nargs="?", help="URL Video YouTube yang akan memproses")
    parser.add_argument("--web", action="store_true", help="Jalankan antarmuka Gradio")
    parser.add_argument("--keep-temp", action="store_true", help="Jangan hapus folder sementara setelah proses selesai (untuk debugging)")
    args = parser.parse_args()

    # Deteksi apakah berjalan di Hugging Face atau user meminta mode web
    if os.getenv("SPACE_ID") or args.web:
        run_web()
        return

    # Default: Run CLI
    run_cli(url=args.url, keep_temp=args.keep_temp)

if __name__ == "__main__":
    main()
