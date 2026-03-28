import gradio as gr
from typing import Callable

from src.domain.interfaces import IUserInterface

class GradioUI(IUserInterface):
    """Implementasi IUserInterface untuk Web (Gradio)."""
    def __init__(self):
        self._log_output = ""

    def print_banner(self) -> None:
        pass # Banner di web biasanya sudah ada di Markdown header

    @property
    def log_output(self) -> str:
        return self._log_output

    def get_input(self, prompt: str) -> str:
        # Di Gradio, input didapat dari event handler (args function), bukan pull-based.
        # Jadi method ini mungkin tidak relevan untuk flow web saat ini,
        # atau bisa return empty jika workflow memaksa minta input.
        return ""

    def get_secure_input(self, prompt: str) -> str:
        return ""

    def show_info(self, msg: str, level: str = "INFO"):
        # Akumulasi log untuk ditampilkan di Textbox Gradio
        self._log_output += f"{msg}\n"

    def show_error(self, msg: str):
        self._log_output += f"❌ ERROR: {msg}\n"

    # Compatibility overrides
    def log(self, msg: str): self.show_info(f"   -> {msg}")
    
    def show_step(self, msg: str): self.show_info(f"\n--- {msg.upper()} ---")

    def create_demo(self, process_fn: Callable, cleanup_fn: Callable, default_api_key: str = "") -> gr.Blocks:
        """Memasukkan logika perakitan blok Gradio ke dalam UI class."""
        with gr.Blocks(title="HSU AI Clipper") as demo:
            gr.Markdown("# 🎬 HSU AI Clipper")
            gr.Markdown("Otomatis buat video shorts dari YouTube menggunakan AI.")
            
            with gr.Row():
                with gr.Column():
                    session_state = gr.State()
                    url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                    api_input = gr.Textbox(label="Gemini API Key", type="password", value=default_api_key)
                    btn = gr.Button("🚀 Mulai Proses", variant="primary")
                    download_file = gr.File(label="Download All Clips (ZIP)", visible=False)
                    cleanup_btn = gr.Button("🧹 Finish & Cleanup Workspace", variant="secondary", visible=False)
                
                with gr.Column():
                    log_display = gr.Textbox(label="Process Logs", interactive=False, lines=10)
                    video_output = gr.Gallery(label="Generated Clips", visible=False)

            btn.click(
                fn=process_fn,
                inputs=[url_input, api_input],
                outputs=[log_display, video_output, download_file, cleanup_btn, session_state]
            )

            cleanup_btn.click(
                fn=cleanup_fn,
                inputs=[session_state],
                outputs=[log_display, video_output, download_file, cleanup_btn]
            )
        
        return demo
