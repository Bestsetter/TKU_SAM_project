FROM python:3.11-slim

# System deps for OpenCV / matplotlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SAM model weights into the image (avoids runtime download timeout)
RUN python -c "from transformers import SamModel, SamProcessor; SamModel.from_pretrained('facebook/sam-vit-base'); SamProcessor.from_pretrained('facebook/sam-vit-base')"

# Copy project files
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
