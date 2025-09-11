# # syntax=docker/dockerfile:1.6
# ARG ULTRA_BASE=ultralytics/ultralytics:8.3.51-jetson-jetpack6
# FROM ${ULTRA_BASE}

# ENV PATH="/usr/src/tensorrt/bin:${PATH}" \
#     DEBIAN_FRONTEND=noninteractive \
#     YOLO_CONFIG_DIR=/tmp/Ultralytics
# WORKDIR /workspace/optimizer

# # Show TRT banner at build (helps detect wrong TRT version/path)
# RUN if [ -x /usr/src/tensorrt/bin/trtexec ]; then \
#       echo "== trtexec banner ==" && /usr/src/tensorrt/bin/trtexec --help 2>&1 | head -n1 || true; \
#     else \
#       echo "trtexec not found in container"; \
#     fi

# # Jetson(JetPack) 환경에서는 CUDA/TensorRT 저장소 추가 및 관련 패키지 설치 불필요
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       build-essential python3-dev libglib2.0-0 libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Python deps — install ONLY from requirements.txt (keeps versions in one place)
# COPY requirements.txt /workspace/optimizer/requirements.txt
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip uninstall -y cuda || true && \
#     python3 -m pip install --no-cache-dir -r requirements.txt && \
#     python3 -m pip uninstall -y opencv-python opencv-contrib-python || true

# # App code
# COPY . /workspace/optimizer/
# ENV PYTHONPATH="/workspace/optimizer:${PYTHONPATH:-}"
# RUN mkdir -p /tmp/Ultralytics && chmod -R 777 /tmp/Ultralytics

# # Guardrail: require TRT v8 on JP6.0 (R36.3)
# RUN printf '%s\n' '#!/usr/bin/env bash' \
# 'set -e' \
# 'B=$(/usr/src/tensorrt/bin/trtexec --help | head -1 || true)' \
# 'echo "$B" | grep -q "TensorRT v8" || { echo "ERROR: Need TensorRT v8.* (JP6.0); got: $B" >&2; exit 42; }' \
# 'exec "$@"' > /usr/local/bin/require_trt8.sh && chmod +x /usr/local/bin/require_trt8.sh

# EXPOSE 5005
# ENTRYPOINT ["require_trt8.sh"]
# CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5005"]


FROM python:3.10-slim

WORKDIR /workspace2/optimizing

# (optional) silence Ultralytics writable-dir warning
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN mkdir -p /tmp/Ultralytics && chmod -R 777 /tmp/Ultralytics

# Add GUI deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH="/workspace2/optimizing:${PYTHONPATH}"
EXPOSE 5005
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5005"]