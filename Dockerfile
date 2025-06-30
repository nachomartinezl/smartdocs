###############################################################################
# smartdocs – GPU image (CUDA 12.3 base, PaddlePaddle 3.1.0 for cu129)
###############################################################################
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

ARG PADDLE_WHL=paddlepaddle-gpu==3.1.0  # cu129 build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FLAGS_fraction_of_gpu_memory_to_use=0.8

# ------------------------------------------------------------------------- #
#  system deps
# ------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget libgl1 libglib2.0-0 poppler-utils \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------- #
#  venv & dependencies
# ------------------------------------------------------------------------- #
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .

RUN pip install -U pip \
 && pip install --no-cache-dir ${PADDLE_WHL} \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
      --trusted-host www.paddlepaddle.org.cn \
 && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------- #
#  application code
# ------------------------------------------------------------------------- #
WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
