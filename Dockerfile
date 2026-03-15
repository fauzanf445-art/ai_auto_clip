FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    gnupg \
    lsb-release \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verifikasi bahwa Node.js terinstall dengan benar (Wajib agar yt-dlp bekerja)
RUN node -v && npm -v

# 3. Security & Environment Setup
RUN useradd -m -u 1000 user
# USER user
# (Komentar: Kita matikan user default saat build agar CMD bisa berjalan sebagai root untuk fix DNS)

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

WORKDIR $HOME/app

# 4. Install Dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Project
COPY --chown=user . .

# 6. Create Directories
RUN mkdir -p Temp Output models/whispermodels models/mpmodels files logs fonts resources/prompts

# Fix permission: Karena kita build sebagai root, pastikan folder milik user 1000
RUN chown -R user:user $HOME

EXPOSE 7860

# Pastikan app.py dijalankan dalam mode web
# Trik: Inject Google DNS ke resolv.conf, lalu switch ke user biasa untuk jalankan app
CMD ["/bin/bash", "-c", "echo 'nameserver 8.8.8.8' > /etc/resolv.conf && su user -c 'python app.py --web'"]