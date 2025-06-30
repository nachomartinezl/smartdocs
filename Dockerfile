###############################################################################
# smartdocs – GPU image (CUDA 11.8, cuDNN 8, Paddle 3.1.0)
###############################################################################
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

# ------------------------------------------------------------------------- #
#  basic env                                                                
# ------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # let Paddle manage VRAM politely (tweak if you have multiple GPUs)
    FLAGS_fraction_of_gpu_memory_to_use=0.8

# ------------------------------------------------------------------------- #
#  system deps                                                              
# ------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget libgl1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------- #
#  Python venv                                                              
# ------------------------------------------------------------------------- #
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# copy & install pinned requirements
COPY requirements.txt .
RUN pip install -U pip \
 && pip install --no-cache-dir ${PADDLE_WHL} \
 && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------- #
#  Pre-warm PaddleOCR weights (optional – shaves first-run latency)          
# ------------------------------------------------------------------------- #
RUN python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(use_gpu=True, show_log=False)   # downloads models once
PY

# ------------------------------------------------------------------------- #
#  application code                                                         
# ------------------------------------------------------------------------- #
WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
