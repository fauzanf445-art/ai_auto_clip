FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 3. Security & Environment Setup
RUN useradd -m -u 1000 user
USER user
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

EXPOSE 7860

# Pastikan app.py dijalankan dalam mode web
CMD ["python", "app.py", "--web"]