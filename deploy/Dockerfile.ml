# Dockerfile for ML training (RunPod / any GPU server)
# Base: official TensorFlow 2.21 image with GPU support
FROM tensorflow/tensorflow:2.21.0-gpu

# Python 3.12 is already included in the TF 2.21 base image
WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is needed for training
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy the code (no data, models or secrets; they are mounted as volumes)
COPY configs/        configs/
COPY models/train_lstm.py  models/train_lstm.py
COPY models/train_gru.py   models/train_gru.py
COPY models/train_both.py  models/train_both.py
COPY utils/          utils/
COPY controller/     controller/
COPY backtest.py     backtest.py
COPY generate_report.py    generate_report.py
COPY generate_report_V_2.py generate_report_V_2.py

# Create the required directories
RUN mkdir -p data/data_base logs models/saved_models

# TensorFlow environment variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Defaults to an interactive shell; pass the run command explicitly:
# docker run --gpus all -v $(pwd)/data:/workspace/data \
#            -v $(pwd)/models:/workspace/models \
#            -e TELEGRAM_BOT_TOKEN=... \
#            nexgen-ml python models/train_lstm.py --config configs/config.yaml
CMD ["/bin/bash"]
