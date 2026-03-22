import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from src.config import AppConfig
from src.infrastructure.ui.console import ConsoleUI
from src.infrastructure.ui.gradio_ui import create_gradio_interface, GradioUI
from src.infrastructure.ui.progress import LogProgressReporter, TqdmProgressReporter
from src.infrastructure.ui.logging_config import TqdmLogger
from src.bootstrap import Bootstrap
from src.container import Container
from src.application.context import SessionContext

def setup_environment(config: AppConfig):
    """Mempersiapkan lingkungan eksekusi."""
    Bootstrap.setup_directories(config)

def run_extract_cookies(config: AppConfig):
    """Menjalankan logika ekstraksi cookies."""
    container = Container(config)
    print("🍪 Memulai ekstraksi cookies...")
    if container.auth_service.extract_cookies_from_browser(config.paths.COOKIE_FILE):
         print(f"✅ Cookies tersimpan di: {config.paths.COOKIE_FILE}")
    else:
         print("❌ Gagal mengekstrak cookies. Pastikan browser tertutup atau login YouTube.")

def run_cli(config: AppConfig, url: Optional[str] = None, keep_temp: bool = False):
    """Menjalankan aplikasi dalam mode CLI."""
    setup_environment(config)
    
    ui = ConsoleUI()
    ui.print_banner()

    load_dotenv(config.paths.ENV_FILE)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = ui.get_secure_input("Masukkan Gemini API Key")
        with open(config.paths.ENV_FILE, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}")

    try:
        container = Container(config, keep_temp=keep_temp)
        
        ctx = SessionContext(
            ui=ui,
            api_key=api_key,
            progress_reporter=TqdmProgressReporter()
        )
        
        container.auth_service.check_and_setup_cookies(config.paths.COOKIE_FILE)

        if not url:
            url = ui.get_input("Masukkan URL Video YouTube")
        
        url = url.split('#')[0].strip()
        if not url:
            ui.show_error("URL menjadi kosong setelah sanitasi. Harap berikan URL video yang valid.")
            return
            
        container.orchestrator.run(ctx, url)

    except KeyboardInterrupt:
        print("\n👋 Dibatalkan pengguna.")

def run_web(config: AppConfig):
    """Menjalankan aplikasi dalam mode Web."""
    print("🌐 Memulai HSU AI Clipper dalam mode Web (Gradio)...")
    setup_environment(config)

    def process_via_web(url, api_key):
        if not url:
            yield "Error: URL YouTube wajib diisi.", None
            return
        if not api_key:
            yield "Error: Gemini API Key wajib diisi.", None
            return


        ui = GradioUI()
        ui.log("🚀 Memulai inisialisasi sistem...")
        yield ui.log_output, None

        try:
            # Logger khusus untuk web (non-verbose file logging + console safe)
            logger = TqdmLogger(config.paths.LOG_FILE)
            progress_reporter = LogProgressReporter(logger)

            container = Container(config, keep_temp=False)
            
            ctx = SessionContext(
                ui=ui,
                api_key=api_key,
                progress_reporter=progress_reporter
            )
            
            container.auth_service.check_and_setup_cookies(config.paths.COOKIE_FILE)
            
            url = url.split('#')[0].strip()
            ui.log(f"Memproses URL: {url}")
            yield ui.log_output, None
            
            container.orchestrator.run(ctx, url)
            
            safe_name = container.provider_service.get_safe_folder_name(url)
            output_folder = config.paths.OUTPUT_DIR / str(safe_name)
            video_files = list(output_folder.glob("final_*.mp4"))
            
            ui.log("✅ Proses Selesai!")
            yield ui.log_output, [str(v) for v in video_files]
            
        except Exception as e:
            ui.show_error(str(e))
            yield ui.log_output, None

    default_api_key = os.getenv("GEMINI_API_KEY", "")
    demo = create_gradio_interface(process_fn=process_via_web, default_api_key=default_api_key)
    
    is_huggingface = os.getenv("SPACE_ID") is not None
    server_name = "0.0.0.0" if is_huggingface else "127.0.0.1"
    
    demo.queue().launch(server_name=server_name, server_port=7860)
