FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Use Xvfb for virtual display to enable OpenGL rendering
CMD xvfb-run -a --server-args="-screen 0 1024x768x24" uvicorn main:app --host 0.0.0.0 --port 8000