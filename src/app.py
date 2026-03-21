import os
import argparse
from src.config import AppConfig
from src import main as engine

# Suppress TensorFlow/MediaPipe C++ logging to keep UI clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="HSU AI Clipper - Automated Video Shorts Generator")
    parser.add_argument("url", nargs="?", help="URL Video YouTube yang akan memproses")
    parser.add_argument("--extract-cookies", action="store_true", help="Ekstrak cookies dari browser lokal")
    parser.add_argument("--web", action="store_true", help="Jalankan antarmuka Gradio")
    parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan log DEBUG di terminal")
    args = parser.parse_args()

    config = AppConfig()

    # Deteksi apakah berjalan di Hugging Face atau user meminta mode web
    if os.getenv("SPACE_ID") or args.web:
        engine.run_web(config)
        return

    # Handle Command: Extract Cookies
    if args.extract_cookies:
        engine.run_extract_cookies(config)
        return

    # Default: Run CLI
    engine.run_cli(config, url=args.url, verbose=args.verbose)

if __name__ == "__main__":
    main()
