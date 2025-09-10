# syntax=docker/dockerfile:1.6  # Dockerfile 버전 명시

# 사용할 베이스 이미지 지정
ARG ULTRA_BASE=ultralytics/ultralytics:8.3.51-jetson-jetpack6

# 위에서 지정한 베이스 이미지로부터 시작
FROM ${ULTRA_BASE}

# bash 쉘을 기본으로 사용
# 기본 sh 대신 bash를 사용하여, AI/딥러닝 환경에서 필요한 bash 기능을 지원
SHELL ["/bin/bash", "-lc"]

# tensorrt 실행파일 경로 추가
# 비대화형 모드로 apt 사용
# YOLO 설정 디렉토리 환경변수
ENV PATH="/usr/src/tensorrt/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# 작업 디렉토리 설정
WORKDIR /workspace4/training

# -------------------------------------------------------------
# 이 RUN 명령은 Jetson Orin 환경에서 TensorRT v8만 설치되도록 보장합니다.
# 1. apt 캐시 mount로 빌드 속도 향상 (BuildKit 기능)
# 2. 필수 패키지 설치 후, libnvinfer8이 있으면 버전 확인해서 8.x만 설치
# 3. apt 캐시와 불필요한 파일 정리로 이미지 용량 최소화
# 4. trtexec 실행파일이 있으면 배너 출력 (설치 확인)
# Jetson Orin은 TensorRT 버전이 매우 중요하므로, 잘못된 버전 설치를 방지하는 안전장치입니다.
# -------------------------------------------------------------
# RUN --mount=type=cache,target=/var/cache/apt \
#     --mount=type=cache,target=/var/lib/apt/lists \
#     set -euo pipefail && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends build-essential python3-dev libglib2.0-0 libgl1 && \
#     # libnvinfer8 패키지가 있으면 버전 확인 후 설치
#     if apt-cache show libnvinfer8 >/dev/null 2>&1; then \
#       cand="$(apt-cache policy libnvinfer8 | sed -n 's/  Candidate: //p')"; \
#       case "$cand" in 8.*) \
#         apt-get install -y --no-install-recommends libnvinfer8 libnvinfer-plugin8 libnvinfer-bin python3-libnvinfer python3-libnvinfer-dev ;; \
#         *) echo "Skipping TRT from APT ($cand)";; \
#       esac; \
#     else echo "No TRT in APT; skipping"; fi && \
#     rm -rf /var/lib/apt/lists/* && \
#     # trtexec 실행파일이 있으면 배너 출력
#     if [ -x /usr/src/tensorrt/bin/trtexec ]; then \
#       echo '== trtexec banner =='; /usr/src/tensorrt/bin/trtexec --help 2>&1 | head -n1 || true; \
#     else \
#       echo 'trtexec not found in container'; \
#     fi

# RUN rm -f /var/lib/apt/lists/lock && rm -f /var/cache/apt/archives/lock

# RUN --mount=type=cache,target=/var/cache/apt \
#     --mount=type=cache,target=/var/lib/apt/lists \
#     set -euo pipefail && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends build-essential python3-dev libglib2.0-0 libgl1 && \
#     if apt-cache show libnvinfer8 >/dev/null 2>&1; then \
#       cand="$(apt-cache policy libnvinfer8 | sed -n 's/  Candidate: //p')"; \
#       case "$cand" in 8.*) \
#         apt-get install -y --no-install-recommends libnvinfer8 libnvinfer-plugin8 libnvinfer-bin python3-libnvinfer python3-libnvinfer-dev ;; \
#         *) echo "Skipping TRT from APT ($cand)";; \
#       esac; \
#     else echo "No TRT in APT; skipping"; fi && \
#     rm -rf /var/lib/apt/lists/* && \
#     if [ -x /usr/src/tensorrt/bin/trtexec ]; then \
#       echo '== trtexec banner =='; /usr/src/tensorrt/bin/trtexec --help 2>&1 | head -n1 || true; \
#     else \
#       echo 'trtexec not found in container'; \
#     fi

## (위 단계에서 배너 출력 및 버전 고정)

# -------------------------------------------------------------
# Python 패키지 의존성 설치 (requirements.txt에서만 설치)
# requirements.txt를 복사해서 버전 관리를 일원화합니다.
# pip 캐시 mount로 빌드 속도 향상, 불필요 패키지/캐시 정리로 이미지 용량 최소화
# -------------------------------------------------------------
COPY requirements.txt /workspace4/training/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
  python3 -m pip install --upgrade pip && \
  python3 -m pip uninstall -y cuda || true && \
  python3 -m pip install --no-cache-dir -r requirements.txt && \
  python3 -m pip uninstall -y opencv-python opencv-contrib-python || true && \
  python3 -m pip cache purge || true

# -------------------------------------------------------------
# 앱 코드 복사 및 환경설정
# 1. 소스코드를 컨테이너에 복사해야 독립 실행 환경에서 앱이 정상 동작
# 2. PYTHONPATH 설정으로 파이썬이 복사된 소스코드를 모듈로 인식
# 3. YOLO 임시 디렉토리 생성 및 권한 설정 (권한 문제 방지)
# -------------------------------------------------------------
COPY . /workspace4/training/
ENV PYTHONPATH="/workspace4/training"
RUN mkdir -p /tmp/Ultralytics && chmod -R 777 /tmp/Ultralytics

# -------------------------------------------------------------
# TensorRT v8 확인 및 실행 스크립트 생성
# 이 RUN 명령은 여러 경로에서 trtexec 실행파일을 찾고, 실제로 TensorRT v8 버전이 설치되어 있는지 확인합니다.
# - v8이 아니면 에러를 내고, 없으면 경고만 출력
# - Jetson Orin 6.0에서 TensorRT v8이 필수이므로, 잘못된 버전(TRT10)/미설치 상태를 미리 감지하여 오류 방지
# - 여러 경로를 확인하는 이유는 Jetson 환경마다 설치 위치가 다를 수 있기 때문
# -------------------------------------------------------------
# -------------------------------------------------------------
# Generate /usr/local/bin/require_trt8.sh (clean heredoc, no quoting headaches)
# -------------------------------------------------------------
# create the script
RUN cat > /usr/local/bin/require_trt8.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

TRT_PATHS=(/usr/src/tensorrt/bin/trtexec /usr/bin/trtexec /usr/local/bin/trtexec)
FOUND=""
for p in "${TRT_PATHS[@]}"; do
  if [ -x "$p" ]; then
    B=$("$p" --help 2>/dev/null | sed -n '1,8p' || true)
    VER=$("$p" --help 2>/dev/null | grep -m1 -o 'TensorRT v[0-9]\+' || true)
    echo "trtexec banner: $B"
    FOUND="$p"
    case "${VER:-}" in
      "TensorRT v8"*) : ;;  # OK
      *) echo "ERROR: Need TensorRT v8.* (JP6.0); got: ${VER:-unknown} at $p" >&2; exit 42 ;;
    esac
    break
  fi
done

if [ -z "$FOUND" ]; then
  echo "WARNING: trtexec not found in any known path; continuing without guard." >&2
fi

exec "$@"
BASH

# normalize line endings + mark executable
RUN sed -i 's/\r$//' /usr/local/bin/require_trt8.sh && chmod +x /usr/local/bin/require_trt8.sh





# TensorRT v8 확인 및 실행 스크립트 생성
# 5005 포트 오픈
EXPOSE 5003
# 컨테이너 시작 시 TensorRT v8 확인 스크립트 실행
ENTRYPOINT ["require_trt8.sh"]
# uvicorn으로 FastAPI 앱 실행
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5003"]