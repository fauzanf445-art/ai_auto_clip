import gradio as gr
from typing import Callable, Any

from src.domain.interfaces import IUserInterface

class GradioUI(IUserInterface):
    """Implementasi IUserInterface untuk Web (Gradio)."""
    def __init__(self):
        self.log_output = ""

    def get_input(self, prompt: str) -> str:
        # Di Gradio, input didapat dari event handler (args function), bukan pull-based.
        # Jadi method ini mungkin tidak relevan untuk flow web saat ini,
        # atau bisa return empty jika workflow memaksa minta input.
        return ""

    def get_secure_input(self, prompt: str) -> str:
        return ""

    def show_info(self, msg: str):
        # Akumulasi log untuk ditampilkan di Textbox Gradio
        self.log_output += f"{msg}\n"

    def show_error(self, msg: str):
        self.log_output += f"❌ ERROR: {msg}\n"

    # Compatibility overrides
    def log(self, msg: str): self.show_info(f"   -> {msg}")
    def show_step(self, msg: str): self.show_info(f"\n--- {msg.upper()} ---")

def create_gradio_interface(process_fn: Callable[[str, str], Any], default_api_key: str = ""):
    """
    Membuat dan mengembalikan objek Gradio Blocks (UI Definition).
    
    Args:
        process_fn: Fungsi callback yang akan dipanggil saat tombol diklik. 
                   Signature: (url, api_key) -> (log_text, video_gallery)
        default_api_key: Nilai default untuk field API Key.
    """
    with gr.Blocks(title="HSU AI Clipper") as demo:
        gr.Markdown("# 🎬 HSU AI Clipper")
        gr.Markdown("Otomatis buat video shorts dari YouTube menggunakan AI.")
        
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                api_input = gr.Textbox(label="Gemini API Key", type="password", value=default_api_key)
                btn = gr.Button("🚀 Mulai Proses", variant="primary")
            
            with gr.Column():
                log_display = gr.Textbox(label="Process Logs", interactive=False, lines=10)
                video_output = gr.Gallery(label="Generated Clips")

        btn.click(
            fn=process_fn,
            inputs=[url_input, api_input],
            outputs=[log_display, video_output]
        )
    
    return demo
