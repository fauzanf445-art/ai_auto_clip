import os
import gradio as gr
import argparse
from typing import Optional

from src.container import Container
from src.application.context import SessionContext
from src.infrastructure.ui.progress import TqdmProgressReporter, LogProgressReporter
from src.infrastructure.ui.logging_config import ContextualLogger
from src.infrastructure.ui.console import ConsoleUI
from src.infrastructure.ui.gradio_ui import GradioUI

# Suppress TensorFlow/MediaPipe C++ logging to keep UI clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_cli(url: Optional[str] = None, clean_temp: bool = False):
    """Menjalankan aplikasi dalam mode CLI."""
    try:
        ui = ConsoleUI()
        ui.print_banner()

        # 1. Setup Environment & Credentials via UI
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            api_key = ui.get_secure_input("Masukkan Gemini API Key")
        
        if not url:
            url = ui.get_input("Masukkan URL Video YouTube")

        url = url.split('#')[0].strip()
        if not url:
            ui.show_error("URL tidak valid.")
            return 

        # 2. Instantiate Container and Context
        container = Container(clean_temp=clean_temp, verbose=False)
        session_logger = ContextualLogger(ui=ui, base_logger=container.logger)
        ctx = SessionContext(
            ui=ui, 
            api_key=api_key, 
            url=url, 
            logger=session_logger,
            progress_reporter=TqdmProgressReporter()
        )

        # 3. Execute
        for update in container.workflow.execute_workflow(url, ctx):
            # Karena workflow sekarang generator, kita harus mengiterasinya agar berjalan.
            if isinstance(update, str):
                ui.show_info(update)

    except KeyboardInterrupt:
        print("\n👋 Dibatalkan pengguna.")

def run_web(clean_temp: bool = False):
    """Menjalankan aplikasi dalam mode Web dengan dukungan penuh Google Colab."""
    print("🌐 Memulai HSU AI Clipper dalam mode Web (Gradio)...")

    # --- Logika Deteksi Environment ---
    is_huggingface = os.getenv("SPACE_ID") is not None
    is_colab = "COLAB_RELEASE_TAG" in os.environ

    clean_temp_global = clean_temp

    def process_via_web(url: str, api_key: str):
        if not url: 
            yield "Error: URL YouTube wajib diisi.", None, None, gr.update(visible=False), None
            return
        if not api_key:
            yield "Error: Gemini API Key wajib diisi.", None, None, gr.update(visible=False), None
            return

        # Inisialisasi Container dan UI untuk setiap request web
        container = Container(clean_temp=clean_temp_global)
        ui = GradioUI()

        ui.log("🚀 Memulai inisialisasi sistem...")
        yield ui.log_output, None, None, gr.update(visible=False), None

        try:
            # Membuat SessionContext dengan url yang diberikan
            ctx = SessionContext(
                ui=ui, 
                api_key=api_key, 
                url=url,
                logger=ContextualLogger(ui=ui, base_logger=container.logger),
                progress_reporter=LogProgressReporter(container.logger)
            )
            
            url_clean = url.split('#')[0].strip()
            ui.log(f"Memproses URL: {url_clean}")
            yield ui.log_output, None, None, gr.update(visible=False), None
            
            # Memanggil execute_workflow (bukan run)
            video_files = []
            for update in container.workflow.execute_workflow(url_clean, ctx):
                if isinstance(update, list):
                    video_files = update
                else:
                    # Jika update adalah string status
                    ui.log(update)
                    yield ui.log_output, None, None, gr.update(visible=False), None
            
            # Buat paket ZIP
            zip_path = container.workflow.prepare_download_package(ctx)
            
            ui.log("✅ Proses Selesai!")
            yield (
                ui.log_output, 
                [str(v) for v in video_files], 
                gr.update(value=str(zip_path) if zip_path else None, visible=True), 
                gr.update(visible=True),
                ctx
            )
            
        except Exception as e:
            ui.show_error(str(e))
            yield ui.log_output, None, None, gr.update(visible=False), None

    def cleanup_session(ctx: Optional[SessionContext]):
        if not ctx:
            return "Tidak ada sesi aktif.", None, gr.update(visible=False), gr.update(visible=False)
        
        # Jalankan cleanup via workflow
        container = Container(clean_temp=False)
        container.workflow.complete_and_cleanup(ctx)
        
        return "🧹 Workspace dibersihkan. Anda bisa menutup tab atau memproses URL baru.", None, gr.update(visible=False), gr.update(visible=False)

    default_api_key = os.getenv("GEMINI_API_KEY", "")
    
    ui_factory = GradioUI()
    demo = ui_factory.create_demo(
        process_fn=process_via_web, 
        cleanup_fn=cleanup_session,
        default_api_key=default_api_key
    )
    
    # Konfigurasi Network
    server_name = "0.0.0.0" if (is_huggingface or is_colab) else "127.0.0.1"
    should_share = True if is_colab else False

    print(f"🔧 Config: Colab={is_colab}, Share={should_share}, Host={server_name}")
    
    if demo is not None:
        demo.queue().launch(
            server_name=server_name, 
            server_port=7860,
            share=should_share,
            debug=is_colab
        )

def main():
    parser = argparse.ArgumentParser(description="HSU AI Clipper - Automated Video Shorts Generator")
    parser.add_argument("url", nargs="?", help="URL Video YouTube yang akan memproses")
    parser.add_argument("--web", action="store_true", help="Jalankan antarmuka Gradio")
    parser.add_argument("--clean-temp", action="store_true", help="Hapus folder sementara")
    args = parser.parse_args()

    if os.getenv("SPACE_ID") or args.web:
        run_web(clean_temp=args.clean_temp)
    else:
        run_cli(url=args.url, clean_temp=args.clean_temp)

if __name__ == "__main__":
    main()
