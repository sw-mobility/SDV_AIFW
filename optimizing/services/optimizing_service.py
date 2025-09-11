############################################################
# 엔진 빌드 동시성 제한 (OOM 방지) 및 최적화 유틸 스크립트
#
# 이 모듈은 모델 최적화(ONNX→TensorRT, INT8 양자화, 프루닝, PT→ONNX)
# 파이프라인을 수행하는 핵심 로직을 담고 있으며, 다음을 제공합니다.
#  - trtexec 기반 FP32/FP16 엔진 빌드 (DLA 지정 및 GPU 폴백)
#  - TensorRT Python API 기반 INT8 엔진 빌드(엔트로피 캘리브레이터)
#  - 산출물 통계(.txt) 생성 및 메트릭 수집
#  - PT/ONNX/ENGINE 파일의 경량 통계 추출
#  - YOLO 모델 프루닝(비구조/구조) 및 PT→ONNX 내보내기
#
# 운영 메모:
#  - 동시에 실행 가능한 엔진 빌드 수를 세마포어로 제한하여 OOM/메모리 경합을 완화합니다.
#  - Jetson Orin에는 DLA 코어가 2개(dla0, dla1) 있어, DLA 작업이 2개를 넘으면
#    일부 작업은 TensorRT의 GPU 폴백이 발생할 수 있습니다(allowGPUFallback).
############################################################

import os
import sys
import glob
import shutil
import logging
import asyncio
from typing import Dict, Any
import shlex
import tensorrt as trt
# ==========================================================
# 동시성 제한 세마포어
# - 동시에 실행할 수 있는 엔진 빌드 개수 제한
# - 환경변수 ENGINE_BUILD_CONCURRENCY로 조정 (기본 3)
#   * Orin DLA는 2개 코어이므로 2 이상 병렬 시 GPU 폴백 증가 가능
# ==========================================================
_BUILD_SEM = asyncio.Semaphore(int(os.getenv("ENGINE_BUILD_CONCURRENCY", "3")))
logger = logging.getLogger("optimizing_service")

# ========= 서드파티 라이브러리 =========
import numpy as np
import onnx
import torch
from fastapi import HTTPException
import torch.nn.utils.prune

# 타이밍 캐시: 같은 ONNX 재빌드 속도를 크게 향상
import hashlib, tempfile
import gc
# ========= 로컬/프로젝트 의존성 =========
# 로컬 ultralytics 우선 사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ultralytics"))
from ultralytics import YOLO

from models.optimizing_model import (
    OnnxToTrtParams,
    OnnxToTrtInt8Params,
    PruneUnstructuredParams,
    PruneStructuredParams,
    PtToOnnxParams,
    CheckModelStatsParams,
)

from PIL import Image

# 영속 타이밍 캐시(재빌드 속도 향상)
import hashlib, tempfile

from onnx import numpy_helper
# ----------------------------------------------------------
# 공통: best.* 기본 출력 경로 강제/생성 도우미
# ----------------------------------------------------------
def _default_out(workdir: str, kind: str) -> str:
    ext = {
        "prune_unstructured": "pt",
        "prune_structured": "pt",
        "pt_to_onnx": "onnx",
        "onnx_to_trt": "engine",
        "onnx_to_trt_int8": "engine",
    }.get(kind, "out")
    return os.path.abspath(os.path.join(workdir or ".", f"best.{ext}"))

def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# trtexec 바이너리 경로 확인(우선 PATH, 없으면 표준 설치 경로)
TRTEXEC = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"

def _ensure_trtexec_available():
    """trtexec 사용 가능 여부를 검증합니다."""
    if not shutil.which("trtexec") and not os.path.exists("/usr/src/tensorrt/bin/trtexec"):
        raise HTTPException(status_code=400, detail="trtexec not found in PATH or /usr/src/tensorrt/bin")
# =============== helpers for defaults ===============
def _default_hw_from_env_or_onnx(onnx_path: str) -> tuple[int, int]:
    # You can extend this to actually read H/W from ONNX if you prefer.
    H = int(os.getenv("TRT_DEFAULT_H", "640"))
    W = int(os.getenv("TRT_DEFAULT_W", "640"))
    return (H, W)

def _default_workspace_mib() -> int:
    return int(os.getenv("TRT_WORKSPACE_MIB", "0")) or (get_max_workspace_bytes() >> 20)


def onnx_to_trt_fp32(onnx_path: str, engine_path: str, device: str = "cuda") -> str:
    """
    Synchronous FP32 build via TensorRT Python API. Intended for non-async callers.
    """
    return _trt_build_sync(
        onnx_path,
        engine_path,
        precision="fp32",
        device=device,
        default_hw=_default_hw_from_env_or_onnx(onnx_path),
        workspace_mib=_default_workspace_mib(),
        timing_cache_dir=os.getenv("TRT_TIMING_CACHE_DIR"),
    )

def onnx_to_trt_fp16(onnx_path: str, engine_path: str, device: str = "cuda") -> str:
    """
    Synchronous FP16 build via TensorRT Python API. Intended for non-async callers.
    """
    return _trt_build_sync(
        onnx_path,
        engine_path,
        precision="fp16",
        device=device,
        default_hw=_default_hw_from_env_or_onnx(onnx_path),
        workspace_mib=_default_workspace_mib(),
        timing_cache_dir=os.getenv("TRT_TIMING_CACHE_DIR"),
    )

# =============== MAIN async API (kept simple & correct) ===============
async def onnx_to_trt(params: OnnxToTrtParams) -> Dict[str, Any]:
    """
    ONNX → TensorRT 엔진 빌드(FP32/FP16, TensorRT Python API).
    - 동시 빌드 수를 세마포어로 제한(OOM 완화)
    - DLA 지정 시 DLA 사용, 미지원/혼잡 시 GPU 폴백 허용
    - 엔진 옆에 <stem>_stats.txt 작성 및 metrics 반환
    """
    _ensure_trt_available()
    onnx_path = params.input_path
    engine_path = params.output_path or _default_out(getattr(params.info, "workdir", "."), "onnx_to_trt")

    if not onnx_path or not engine_path:
        raise HTTPException(status_code=400, detail="input_path and output_path are required")
    if not os.path.isfile(onnx_path):
        raise HTTPException(status_code=400, detail=f"ONNX not found: {onnx_path}")
    if params.device == "cpu":
        raise HTTPException(status_code=400, detail="TensorRT does not support CPU engines")

    logger.info(
        f"[ONNX->TRT] TRT API build start | input={onnx_path} "
        f"output={engine_path} device={params.device} precision={params.precision}"
    )

    # One global semaphore is enough; make sure it's defined once in the module.
    async with _BUILD_SEM:
        # Use the async wrapper that already offloads to a worker thread.
        built_path = await optimize_onnx_to_engine(
            onnx_path,
            engine_path=engine_path,
            precision=params.precision.lower(),
            device=params.device,
            default_hw=_default_hw_from_env_or_onnx(onnx_path),
            workspace_mib=_default_workspace_mib(),
            reset_cuda=(os.getenv("TRT_RESET_CUDA", "1") == "1"),
        )
    # Write stats + metrics
    out_dir = os.path.dirname(built_path) or "."
    op_name = f"onnx_to_trt_{params.precision.lower()}"
    stats_txt = await write_stats_txt(op=op_name, model_path=built_path, out_dir=out_dir)
    stats = await compute_model_stats(built_path)

    metrics = {
        "kind": "onnx_to_trt",
        "precision": params.precision.lower(),
        "device": params.device,
        "stats": stats,
        "artifact_files": [os.path.basename(built_path), os.path.basename(stats_txt)],
        **_metrics_for_artifact(built_path),
    }
    return {"artifacts_path": out_dir, "metrics": metrics}

# ==========================================================
# 보조 유틸: 이미지 나열/전처리 (INT8 캘리브레이션용)
# ==========================================================
def _list_images(d: str):
    """디렉토리에서 지원 확장자의 이미지를 정렬하여 반환합니다.""" # from PIL import Image
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files.sort()
    return files

def _preprocess(im_path: str, size_hw):
    """PIL로 이미지를 로드하고 [0,1] 정규화 후 NCHW(float32)로 변환합니다."""
    H, W = size_hw  # (H, W) = (Height, Width)
    Resample = getattr(Image, "Resampling", Image)
    img = Image.open(im_path).convert("RGB").resize((W, H), Resample.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr

# ==========================================================
# ONNX → TensorRT INT8 (TensorRT Python API)
# - 엔트로피 캘리브레이터 사용
# - 트라이밍 캐시/프로파일/FP16 혼합/스파스 등 옵션 지원
# ==========================================================
async def onnx_to_trt_int8(params: OnnxToTrtInt8Params) -> Dict[str, Any]:
    """
    INT8 TensorRT 엔진 빌드(엔트로피 캘리브레이터).
    반환: {artifacts_path, metrics} 및 엔진 옆에 stats .txt 작성.
    """
    _ensure_trt_available()
    try:
        # 최신 cuda-python 계열
        from cuda.bindings import runtime as cudart
    except Exception:
        # 구버전/대체 모듈
        from cuda import cudart

    # ---- 사용자 정의 캘리브레이터 ----
    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """간단한 엔트로피 캘리브레이터 구현(Host→Device 복사 포함)."""
        def _rc_ok(self, rc):
            # cuda 에러코드 판별(버전별 타입 차이를 안전하게 처리)
            try:
                if isinstance(rc, tuple):
                    rc = rc[0]
                return rc == cudart.cudaError_t.cudaSuccess
            except Exception:
                try:
                    return int(rc) == 0
                except Exception:
                    return False

        def __init__(self, img_paths, batch_size, input_shape, cache_path):
            super().__init__()
            # 배치 정렬(남는 이미지는 버림)
            n_imgs = len(img_paths) - (len(img_paths) % batch_size)
            self.imgs = img_paths[:n_imgs] if n_imgs > 0 else img_paths
            self.batch_size = batch_size
            self.input_shape = input_shape  # (N,C,H,W)
            self.cache_path = cache_path
            self.current = 0
            self.device_input = None
            self.nbytes = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
            logger.info(f"INT8 Calibrator: {len(self.imgs)} images, bs={self.batch_size}, cache={self.cache_path}")

        def __del__(self):
            # 디바이스 메모리 정리(가능한 범위에서)
            try:
                if self.device_input:
                    _ = cudart.cudaFree(self.device_input)
            except Exception:
                pass

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            # 더 이상 데이터가 없으면 None 반환 → 캘리브레이션 종료
            if self.current >= len(self.imgs):
                return None
            bs = min(self.batch_size, len(self.imgs) - self.current)
            N, C, H, W = self.input_shape
            batch = np.zeros((bs, C, H, W), dtype=np.float32)
            for i, p in enumerate(self.imgs[self.current:self.current+bs]):
                batch[i] = _preprocess(p, (H, W))
            self.current += bs

            # 최초 호출 시 디바이스 버퍼 할당
            if self.device_input is None:
                rc, dptr = cudart.cudaMalloc(self.nbytes)
                if not self._rc_ok(rc):
                    raise RuntimeError(f"cudaMalloc failed: {rc}")
                self.device_input = dptr

            # Host→Device 복사
            rc = cudart.cudaMemcpy(
                self.device_input,
                batch.ctypes.data,
                batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            if not self._rc_ok(rc):
                raise RuntimeError(f"cudaMemcpy HtoD failed: {rc}")
            return [int(self.device_input)]

        def read_calibration_cache(self):
            # 기존 캐시가 있으면 재사용(빌드 시간 단축)
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    data = f.read()
                logger.info(f"Loaded existing INT8 cache: {self.cache_path}")
                return data
            return None

        def write_calibration_cache(self, cache):
            # 새로 생성된 캐시 저장
            with open(self.cache_path, "wb") as f:
                f.write(cache)
            logger.info(f"Wrote INT8 cache: {self.cache_path}")

    onnx_path = params.input_path
    engine_path = params.output_path or _default_out(getattr(params.info, "workdir", "."), "onnx_to_trt_int8")
    calib_dir = params.calib_dir

    # 필수 파라미터 방어적 체크
    if not onnx_path or not engine_path or not calib_dir:
        raise HTTPException(status_code=400, detail="input_path, output_path, and calib_dir are required")
    if params.device == "cpu":
        raise HTTPException(status_code=400, detail="TensorRT does not support CPU engines")

    # 동기 빌드 함수 (스레드 오프로딩용)
    def _build_int8_sync() -> str:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(flag)
        parser  = trt.OnnxParser(network, TRT_LOGGER)

        # ONNX 파싱
        if not os.path.isfile(onnx_path):
            raise HTTPException(status_code=400, detail=f"ONNX not found: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = "\n".join([str(parser.get_error(i)) for i in range(parser.num_errors)])
                raise HTTPException(status_code=400, detail=f"ONNX parse failed:\n{errs}")

        # 빌더 설정(워크스페이스/타이밍 캐시)
        config = builder.create_builder_config()
        ws_mib = int(getattr(params, "workspace_mib", 0) or (get_max_workspace() >> 20))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mib * (1 << 20))

        cache_dir = os.getenv("TRT_TIMING_CACHE_DIR", tempfile.gettempdir())
        os.makedirs(cache_dir, exist_ok=True)
        tc_path = os.path.join(
            cache_dir, f"trt_timing_{hashlib.sha1(os.path.abspath(onnx_path).encode()).hexdigest()[:10]}.blob"
        )
        try:
            if hasattr(trt, "TimingCache"):
                try:
                    with open(tc_path, "rb") as f:
                        cache = trt.TimingCache(f.read(), TRT_LOGGER)
                except Exception:
                    cache = trt.TimingCache(TRT_LOGGER)
                config.set_timing_cache(cache, ignore_mismatch=True)
        except Exception:
            pass

        # 최적화 프로파일 설정(배치=1 고정, 동적 공간에 대해 min/opt/max)
        H, W = params.input_size  # (H, W) = (Height, Width)
        in0 = network.get_input(0)
        dims = tuple(in0.shape)  # 예: (1, 3, 640, 640) 또는 (-1, 3, -1, -1)
        def _is_static_4d(d):
            return (len(d) == 4) and all(isinstance(x, int) and x > 0 for x in d)
        profile = builder.create_optimization_profile()

        if _is_static_4d(dims):
            min_shape = opt_shape = max_shape = dims
        else:
            min_shape = (1, 3, 320, 320)
            opt_shape = (1, 3, H, W)
            max_shape = (1, 3, 1280, 1280)

        profile.set_shape(in0.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        if hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)

        # 플래그: INT8(+선택 FP16 혼합), Sparse 지원
        config.set_flag(trt.BuilderFlag.INT8)
        if params.mixed_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if params.sparse and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # DLA 설정: 'dla', 'dla0', 'dla1' 허용 (없으면 무시)
        if params.device.startswith("dla") and hasattr(trt, "DeviceType"):
            config.default_device_type = trt.DeviceType.DLA
            try:
                core = int(params.device[-1]) if params.device in ("dla0", "dla1") else 0
                if hasattr(config, "DLA_core"):
                    config.DLA_core = core
            except Exception:
                pass
            if hasattr(trt.BuilderFlag, "GPU_FALLBACK"):
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # 캘리브레이터 준비(배치 1, opt shape과 동일한 크기)
        cache_path = os.path.join(os.path.dirname(engine_path) or ".", "calibration.cache")
        imgs = _list_images(calib_dir)
        if not imgs:
            raise HTTPException(status_code=400, detail=f"No images found in calib_dir: {calib_dir}")
        BATCH = 1
        max_imgs = max(1, params.int8_max_batches * BATCH)
        n_imgs = max_imgs - (max_imgs % BATCH)
        imgs = imgs[:n_imgs] if n_imgs > 0 else imgs[:max_imgs]
        calib_H, calib_W = opt_shape[2], opt_shape[3]
        calibrator = _EntropyCalibrator(imgs, batch_size=BATCH, input_shape=(BATCH, 3, calib_H, calib_W), cache_path=cache_path)
        config.int8_calibrator = calibrator

        # 엔진 빌드 및 저장
        engine = builder.build_engine(network, config)
        if engine is None:
            raise HTTPException(status_code=500, detail="INT8 engine build returned None")

        # 타이밍 캐시 갱신 저장(가능 시)
        try:
            if hasattr(config, "get_timing_cache"):
                ser = config.get_timing_cache().serialize()
                with open(tc_path, "wb") as f:
                    f.write(ser)
        except Exception:
            pass

        _ensure_parent(engine_path)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        logger.info(f"INT8 engine saved: {engine_path}")
        return engine_path

    # 동시성 제한(세마포어) 하에 빌드 수행 → OOM/경합 완화
    async with _BUILD_SEM:
        engine_out = await asyncio.to_thread(_build_int8_sync)

    # 산출물 경로/통계/메트릭 정리
    out_dir = os.path.dirname(engine_out) or "."
    stats_txt = await write_stats_txt(op="onnx_to_trt_int8", model_path=engine_out, out_dir=out_dir)
    stats = await compute_model_stats(engine_out)
    metrics = {
        "kind": "onnx_to_trt_int8",
        "precision": "int8",
        "device": params.device,
        "mixed_fp16": params.mixed_fp16,
        "sparse": params.sparse,
        "int8_max_batches": params.int8_max_batches,
        "stats": stats,
        "artifact_files": [os.path.basename(engine_out), os.path.basename(stats_txt)],
        **_metrics_for_artifact(engine_out),
    }
    return {"artifacts_path": out_dir, "metrics": metrics}
# ==========================================================
# 공통 유틸 함수들
# ==========================================================
def get_device() -> str:
    """가용 디바이스 반환: 'cuda' 또는 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def _bytes_to_mb(b: int) -> float:
    """바이트를 MB 단위로 변환(소수 2자리 반올림)."""
    try:
        return round(b / 1024 / 1024, 2)
    except Exception:
        return 0.0

def _write_text_file(path: str, text: str) -> None:
    """텍스트 파일 저장(UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ==========================================================
# Stats helpers: .pt / .onnx / .engine 경량 통계 수집
# - .engine 은 TensorRT 메타데이터(가능 시) 포함
# ==========================================================
def _compute_model_stats_sync(model_path: str) -> Dict[str, Any]:
    """
    모델 파일 종류별 경량 통계 계산.
     - .pt/.onnx: 파라미터 수(total/nonzero), 희소도 등
     - .engine : 파일 크기 + IO 텐서/메모리 등(TRT 메타데이터)
    """
    stats: Dict[str, Any] = {}
    stats["path"] = os.path.abspath(model_path)
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pt":
        stats["format"] = "pt"
    elif ext == ".onnx":
        stats["format"] = "onnx"
    elif ext == ".engine":
        stats["format"] = "engine"
    else:
        stats["format"] = "other"

    stats["size_mb"] = _bytes_to_mb(os.path.getsize(model_path))
    total = 0
    nonzero = 0
    if stats["format"] == "pt":
        logger.info("Computing stats for .pt model: %s", model_path)
        y = YOLO(model_path)
        model = y.model.to("cpu").eval()
        with torch.no_grad():
            for p in model.parameters():
                total += p.numel()
                nonzero += torch.count_nonzero(p).item()

    elif stats["format"] == "onnx":
        logger.info("Computing stats for .onnx model: %s", model_path)
        model = onnx.load(model_path, load_external_data=True)
        for tensor in model.graph.initializer:
            try:
                arr = numpy_helper.to_array(tensor)
            except Exception:
                arr = np.array([])
            total += arr.size
            nonzero += np.count_nonzero(arr)

    elif stats["format"] == "engine":
        # TensorRT 메타데이터 시도(가용 시)
        md = _trt_engine_metadata(model_path)
        if md:
            stats["tensorrt_version"] = md.get("tensorrt_version")
            stats["num_io_tensors"] = md.get("num_io_tensors")
            stats["num_inputs"] = md.get("num_inputs")
            stats["num_outputs"] = md.get("num_outputs")
            stats["inputs"] = md.get("inputs")    # [{name, dtype, shape}, ...]
            stats["outputs"] = md.get("outputs")
            if "device_memory_bytes" in md:
                stats["device_memory_mb"] = round(md["device_memory_bytes"] / 1024 / 1024, 2)

    # ENGINE/기타 포맷은 파라미터 카운트를 0으로 유지
    stats["total_params"] = int(total)
    stats["nonzero_params"] = int(nonzero)
    stats["sparsity_pct"] = round(100.0 * (1 - (nonzero / total)), 2) if total > 0 else 0.0
    return stats

async def compute_model_stats(model_path: str) -> Dict[str, Any]:
    """동기 통계 함수를 스레드로 오프로딩."""
    return await asyncio.to_thread(_compute_model_stats_sync, model_path)

def _format_stats_txt(op: str, model_path: str, stats: Dict[str, Any]) -> str:
    """stats 딕셔너리를 사람이 읽기 쉬운 txt 포맷으로 변환."""
    lines = [
        f"operation: {op}",
        f"path: {os.path.abspath(model_path)}",
        f"format: {stats.get('format')}",
        f"size_mb: {stats.get('size_mb')}",
        f"total_params: {stats.get('total_params')}",
        f"nonzero_params: {stats.get('nonzero_params')}",
        f"sparsity_pct: {stats.get('sparsity_pct')}",
    ]

    if stats.get("format") == "engine":
        # TRT 메타데이터(있을 때만)
        if stats.get("tensorrt_version"):
            lines.append(f"tensorrt_version: {stats['tensorrt_version']}")
        if stats.get("device_memory_mb") is not None:
            lines.append(f"device_memory_mb: {stats['device_memory_mb']}")
        if stats.get("num_inputs") is not None:
            lines.append(f"num_inputs: {stats['num_inputs']}")
        if stats.get("num_outputs") is not None:
            lines.append(f"num_outputs: {stats['num_outputs']}")
        if isinstance(stats.get("inputs"), list):
            lines.append("inputs:")
            for i, t in enumerate(stats["inputs"]):
                lines.append(f"  - [{i}] name={t.get('name')} dtype={t.get('dtype')} shape={t.get('shape')}")
        if isinstance(stats.get("outputs"), list):
            lines.append("outputs:")
            for i, t in enumerate(stats["outputs"]):
                lines.append(f"  - [{i}] name={t.get('name')} dtype={t.get('dtype')} shape={t.get('shape')}")

    lines.append("")  # 마지막 줄바꿈
    return "\n".join(lines)

async def write_stats_txt(op: str, model_path: str, out_dir: str) -> str:
    """모델 옆에 <stem>_stats.txt 파일을 생성합니다."""
    os.makedirs(out_dir, exist_ok=True)
    stats = await compute_model_stats(model_path)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    txt_name = f"{stem}_stats.txt"
    txt_path = os.path.abspath(os.path.join(out_dir, txt_name))
    await asyncio.to_thread(_write_text_file, txt_path, _format_stats_txt(op, model_path, stats))
    logger.info("Stats written to %s", txt_path)
    return txt_path

def get_max_workspace() -> int:
    """
    트릭: 현재 GPU에서 사용 가능한 메모리의 90%를 워크스페이스 상한으로 추정.
    실패 시 1GiB로 폴백.
    """
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free * 0.9)
    except Exception as e:
        logger.warning("[Workspace] Could not query GPU memory: %s", e)
    return 1 << 30

# ==========================================================
# TensorRT 관련 보조
# ==========================================================
def _ensure_trt_available():
    """이미지에 TensorRT가 설치되어 있는지 확인."""
    try:
        import tensorrt as trt
    except Exception:
        raise HTTPException(status_code=400, detail="TensorRT is not available in this image")

def _metrics_for_artifact(path: str) -> Dict[str, Any]:
    """산출물 파일의 크기(바이트)를 metrics에 첨부."""
    m: Dict[str, Any] = {}
    try:
        if path and os.path.exists(path):
            m["artifact_size_bytes"] = os.path.getsize(path)
    except Exception:
        pass
    return m

# ==========================================================
# TensorRT 엔진 메타데이터(선택적)
# - 환경변수 TRT_METADATA=0 이면 비활성화
# - 엔진을 역직렬화하여 IO 텐서 정보 등을 추출(가능 시)
# ==========================================================
def _trt_engine_metadata(engine_path: str) -> Dict[str, Any]:
    """
    TensorRT 엔진을 역직렬화하여 IO 메타데이터를 추출합니다.
    TRT 미가용/실패/비활성화 시 {} 반환.
    """
    if os.environ.get("TRT_METADATA", "1") == "0":
        return {}
    try:
        import tensorrt as trt  # 선택 의존성
    except Exception:
        return {}

    try:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            return {}

        # IO 텐서 수집(상한 16개)
        inputs, outputs = [], []
        MAX_IO_LIST = 16
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = list(engine.get_tensor_shape(name))  # JSON 직렬화 친화적
            dtype = str(engine.get_tensor_dtype(name))
            entry = {"name": name, "dtype": dtype, "shape": shape}
            is_input = (hasattr(mode, "name") and mode.name == "INPUT") or (mode == trt.TensorIOMode.INPUT)
            (inputs if is_input else outputs).append(entry)
        inputs = inputs[:MAX_IO_LIST]
        outputs = outputs[:MAX_IO_LIST]

        md = {
            "tensorrt_version": getattr(trt, "__version__", "unknown"),
            "num_io_tensors": engine.num_io_tensors,
            "num_inputs": len(inputs),
            "num_outputs": len(outputs),
            "inputs": inputs,
            "outputs": outputs,
        }
        # 일부 TRT 버전에서 디바이스 메모리 사이즈 제공
        try:
            md["device_memory_bytes"] = int(engine.device_memory_size)
        except Exception:
            pass

        return md
    except Exception:
        return {}

# ==========================================================
# TensorRT 래퍼(공통): FP32/FP16 빌드 + stats.txt 작성
# ==========================================================
async def onnx_to_trt(params: OnnxToTrtParams) -> Dict[str, Any]:
    """
    ONNX → TensorRT 엔진 빌드(FP32/FP16, trtexec).
    - 세마포어로 동시 빌드 수 제한(메모리 경합 방지)
    - DLA 지정 시 DLA 사용, 미지원/혼잡 시 자동 GPU 폴백 허용
    - 엔진 옆에 <stem>_stats.txt 작성 및 metrics 반환
    """
    _ensure_trtexec_available()
    onnx_path = params.input_path
    # 기본값 보정: output 미지정 시 best.engine으로
    engine_path = params.output_path or _default_out(getattr(params.info, "workdir", "."), "onnx_to_trt")
    if not onnx_path or not engine_path:
        raise HTTPException(status_code=400, detail="input_path and output_path are required")

    if params.device == "cpu":
        raise HTTPException(status_code=400, detail="TensorRT does not support CPU engines")
    # 동시성 제한: 여러 엔진 빌드가 겹칠 때 OOM 방지
    async with _BUILD_SEM:
        if params.precision == "fp32":
            logger.info(f"[ONNX->TRT] Running trtexec fp32...")
            await asyncio.to_thread(onnx_to_trt_fp32, onnx_path, engine_path, params.device)
            op_name = "onnx_to_trt_fp32"
        elif params.precision == "fp16":
            logger.info(f"[ONNX->TRT] Running trtexec fp16...")
            await asyncio.to_thread(onnx_to_trt_fp16, onnx_path, engine_path, params.device)
            op_name = "onnx_to_trt_fp16"
        else:
            logger.error(f"[ONNX->TRT] Unsupported precision: {params.precision}")
            raise HTTPException(status_code=400, detail=f"Unsupported precision: {params.precision}")

    logger.info(f"[ONNX->TRT] Conversion finished. Engine path: {engine_path}")

    # 빌드 결과에 대한 stats 파일 작성 및 메트릭 수집
    out_dir = os.path.dirname(engine_path) or "."
    stats_txt = await write_stats_txt(op=op_name, model_path=engine_path, out_dir=out_dir)
    logger.info(f"[ONNX->TRT] Stats written to: {stats_txt}")
    stats = await compute_model_stats(engine_path)
    logger.info(f"[ONNX->TRT] Model stats computed.")

    metrics = {
        "kind": "onnx_to_trt",
        "precision": params.precision,
        "device": params.device,
        "stats": stats,
        "artifact_files": [os.path.basename(engine_path), os.path.basename(stats_txt)],
        **_metrics_for_artifact(engine_path),
    }
    logger.info(f"[ONNX->TRT] Metrics: {metrics}")
    return {"artifacts_path": out_dir, "metrics": metrics}

# ==========================================================
# 코어 작업: 통계/프루닝/PT→ONNX
# ==========================================================
async def check_model_stats(params: CheckModelStatsParams):
    """모델 통계만 산출하여 <stem>_stats.txt 작성."""
    model_path = params.input_path
    logger.info("Entering check_model_stats with model_path: %s", model_path)
    out_dir = (
        os.path.join(params.info.workdir, "output")
        if getattr(params, "info", None)
        else os.path.dirname(model_path)
    )
    os.makedirs(out_dir, exist_ok=True)
    txt_path = await write_stats_txt("check_model_stats", model_path, out_dir)
    stats = await compute_model_stats(model_path)
    metrics = {
        "kind": "check_model_stats",
        "stats": stats,
        "artifact_files": [os.path.basename(txt_path)],
    }
    return {"artifacts_path": out_dir, "metrics": metrics, "stats_txt": txt_path}

async def prune_unstructured_pt(params: PruneUnstructuredParams):
    """
    비구조적 프루닝(가중치 개별 제거: L1 또는 Random).
    - Conv/Linear 전역 프루닝 → 마스크 적용 후 remove로 파라미터 고정
    """
    pt_path = os.path.abspath(params.input_path)
    workdir = getattr(params.info, "workdir", os.path.dirname(pt_path))
    output_path = os.path.abspath(params.output_path or _default_out(workdir, "prune_unstructured"))
    device = get_device()

    pruned_params = await asyncio.to_thread(
        prune_unstructured_sync,
        pt_path,
        output_path,
        device,
        params.pruning_type,
        params.amount
    )
    out_dir = os.path.dirname(output_path)
    txt_path = await write_stats_txt("prune_unstructured", output_path, out_dir)
    stats = await compute_model_stats(output_path)
    metrics = {
        "kind": "prune_unstructured",
        "pruned_params_estimate": pruned_params,
        "stats": stats,
        "artifact_files": [os.path.basename(output_path), os.path.basename(txt_path)],
    }
    return {"artifacts_path": out_dir, "metrics": metrics, "stats_txt": txt_path}

def prune_unstructured_sync(pt_path, output_path, device, pruning_type, amount):
    yolo = YOLO(pt_path)
    model = yolo.model.to(device)
    modules_to_prune = []
    total_params = 0
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            modules_to_prune.append((module, "weight"))
            total_params += module.weight.numel()
    pruning_methods = {
        "l1_unstructured": torch.nn.utils.prune.L1Unstructured,
        "random_unstructured": torch.nn.utils.prune.RandomUnstructured,
    }
    pm = pruning_methods.get(pruning_type, torch.nn.utils.prune.L1Unstructured)
    torch.nn.utils.prune.global_unstructured(modules_to_prune, pruning_method=pm, amount=amount)
    pruned_params_est = int(total_params * amount)
    for module, _ in modules_to_prune:
        torch.nn.utils.prune.remove(module, "weight")
    model = model.to("cpu")
    yolo.model = model
    _ensure_parent(output_path)
    yolo.save(output_path)
    return pruned_params_est

def prune_structured_sync(pt_path, output_path, device, n, dim, amount):
    yolo = YOLO(pt_path)
    model = yolo.model.to(device)
    total_params = 0
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            total_params += module.weight.numel()
    pruned_params = 0
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.utils.prune.ln_structured(
                module, name="weight", amount=amount, n=n, dim=dim
            )
            torch.nn.utils.prune.remove(module, "weight")
            pruned_params += int(module.weight.numel() * amount)
    model = model.to("cpu")
    yolo.model = model
    _ensure_parent(output_path)
    yolo.save(output_path)
    return pruned_params

async def prune_structured_pt(params: PruneStructuredParams):
    pt_path = os.path.abspath(params.input_path)
    workdir = getattr(params.info, "workdir", os.path.dirname(pt_path))
    output_path = os.path.abspath(params.output_path or _default_out(workdir, "prune_structured"))
    device = get_device()
    logger.info("[Pruning] Device used: %s", device)

    pruned_params = await asyncio.to_thread(
        prune_structured_sync,
        pt_path,
        output_path,
        device,
        params.n,
        params.dim,
        params.amount
    )
    out_dir = os.path.dirname(output_path)
    txt_path = await write_stats_txt("prune_structured", output_path, out_dir)
    stats = await compute_model_stats(output_path)
    metrics = {
        "kind": "prune_structured",
        "pruned_params_estimate": pruned_params,
        "stats": stats,
        "artifact_files": [os.path.basename(output_path), os.path.basename(txt_path)],
    }
    return {"artifacts_path": out_dir, "metrics": metrics, "stats_txt": txt_path}

async def pt_to_onnx_fp32(params: PtToOnnxParams):
    """
    PT → ONNX FP32 내보내기.
    - ultralytics.YOLO.export 사용(opset=12, simplify=True)
    - 기본 출력은 workdir/best.onnx
    """
    pt_path = os.path.abspath(params.input_path)
    workdir = getattr(params.info, "workdir", os.path.dirname(pt_path))
    target_path = os.path.abspath(params.output_path or _default_out(workdir, "pt_to_onnx"))
    base = os.path.splitext(pt_path)[0]
    tmp_onnx = base + ".onnx"
    # Run export in a thread
    await asyncio.to_thread(
        YOLO(pt_path).export,
        format="onnx",
        imgsz=params.input_size,
        batch=params.batch_size,
        simplify=True,
        opset=12,
        half=False
    )
    torch.cuda.empty_cache()
    _ensure_parent(target_path)
    if os.path.abspath(tmp_onnx) != os.path.abspath(target_path):
        shutil.copy(tmp_onnx, target_path)
        try:
            os.remove(tmp_onnx)
        except Exception:
            pass

    out_dir = os.path.dirname(target_path)
    txt_path = await write_stats_txt("pt_to_onnx_fp32", target_path, out_dir)
    stats = await compute_model_stats(target_path)
    metrics = {
        "kind": "pt_to_onnx",
        "precision": "fp32",
        "stats": stats,
        "artifact_files": [os.path.basename(target_path), os.path.basename(txt_path)],
    }

    gc.collect()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from cuda import cudart
        cudart.cudaDeviceReset()
        logger.info("Called cudart.cudaDeviceReset()")
    except Exception:
        logger.info("Failed to call cudart.cudaDeviceReset()")

    return {"artifacts_path": out_dir, "metrics": metrics, "stats_txt": txt_path}

async def pt_to_onnx_fp16(params: PtToOnnxParams):
    """
    PT → ONNX FP16 내보내기(half=True).
    - ultralytics.YOLO.export 사용(opset=12, simplify=True, half=True)
    - 기본 출력은 workdir/best.onnx
    """
    pt_path = os.path.abspath(params.input_path)
    workdir = getattr(params.info, "workdir", os.path.dirname(pt_path))
    target_path = os.path.abspath(params.output_path or _default_out(workdir, "pt_to_onnx"))
    base = os.path.splitext(pt_path)[0]
    tmp_onnx = base + ".onnx"
    # Run export in a thread
    await asyncio.to_thread(
        YOLO(pt_path).export,
        format="onnx",
        imgsz=params.input_size,
        batch=params.batch_size,
        simplify=True,
        opset=12,
        half=True
    )
    torch.cuda.empty_cache()
    _ensure_parent(target_path)
    if os.path.abspath(tmp_onnx) != os.path.abspath(target_path):
        shutil.copy(tmp_onnx, target_path)
        try:
            os.remove(tmp_onnx)
        except Exception:
            pass

    out_dir = os.path.dirname(target_path)
    txt_path = await write_stats_txt("pt_to_onnx_fp16", target_path, out_dir)
    stats = await compute_model_stats(target_path)
    metrics = {
        "kind": "pt_to_onnx",
        "precision": "fp16",
        "stats": stats,
        "artifact_files": [os.path.basename(target_path), os.path.basename(txt_path)],
    }

    gc.collect()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from cuda import cudart
        cudart.cudaDeviceReset()
        logger.info("Called cudart.cudaDeviceReset()")
    except Exception:
        logger.info("Failed to call cudart.cudaDeviceReset()")

    return {"artifacts_path": out_dir, "metrics": metrics, "stats_txt": txt_path}

# ==========================================================
# 선택 래퍼 & 정리
# ==========================================================
async def handle_prune_unstructured(params: PruneUnstructuredParams):
    """비구조적 프루닝 래퍼(호출부 가독성용)."""
    ret = await prune_unstructured_pt(params)
    logger.info("Unstructured pruning completed")
    return ret

async def handle_prune_structured(params: PruneStructuredParams):
    """구조적 프루닝 래퍼(호출부 가독성용)."""
    ret = await prune_structured_pt(params)
    logger.info("Structured pruning completed")
    return ret

async def handle_pt_to_onnx(params: PtToOnnxParams):
    """PT→ONNX (기본 FP32) 래퍼(호출부 가독성용)."""
    ret = await pt_to_onnx_fp32(params)
    logger.info("PT to ONNX conversion completed")
    return ret

async def handle_check_model_stats(params: CheckModelStatsParams):
    """모델 통계 산출 래퍼(호출부 가독성용)."""
    ret = await check_model_stats(params)
    logger.info("Model stats check completed")
    return ret

async def cleanup_workdir(workdir: str):
    """작업 디렉터리 정리(실패 무시하고 로그만 남김)."""
    try:
        if os.path.exists(workdir):
            await asyncio.to_thread(shutil.rmtree, workdir)
            logger.info("Cleaned up workdir: %s", workdir)
    except Exception as e:
        logger.error("Failed to clean up workdir %s: %s", workdir, e)

def get_max_workspace_bytes() -> int:
    """Query free GPU memory via cuda-python; fall back to 1 GiB."""
    try:
        from cuda import cudart
        rc, free, total = cudart.cudaMemGetInfo()
        ok = (isinstance(rc, tuple) and rc[0] == 0) or (isinstance(rc, int) and rc == 0)
        if ok and isinstance(free, int):
            return int(free * 0.9)
    except Exception:
        pass
    return 1 << 30  # 1 GiB

def parse_input_info(onnx_path: str) -> tuple[str, bool]:
    """Return (input_name, is_dynamic_4d)."""
    try:
        m = onnx.load(onnx_path)
        inputs = m.graph.input
        four_d = [vi for vi in inputs if vi.type.tensor_type.shape and len(vi.type.tensor_type.shape.dim) == 4]
        in_name = (four_d[0].name if four_d else (inputs[0].name if inputs else "images")) or "images"
        dims = (four_d[0].type.tensor_type.shape.dim if four_d else [])
        def _dimv(d): return d.dim_value if getattr(d, "dim_value", 0) > 0 else -1
        dvals = tuple(_dimv(d) for d in (dims or []))
        is_dynamic = not (len(dvals) == 4 and all(v > 0 for v in dvals))
        return in_name, is_dynamic
    except Exception:
        return "images", True

def maybe_reset_cuda_context():
    """Optional: drop any stale CUDA context *before* building TRT."""
    try:
        from cuda import cudart
        cudart.cudaDeviceReset()
    except Exception:
        pass

# ---------- sync builder (runs inside a worker thread) ----------

def _trt_build_sync(
    onnx_path: str,
    engine_path: str,
    *,
    precision: str = "fp32",        # "fp16" or "fp32"
    device: str = "cuda",           # "cuda" | "dla" | "dla0" | "dla1"
    default_hw: tuple[int, int] = (640, 640),  # (H, W) for opt profile
    workspace_mib: int | None = None,
    timing_cache_dir: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPL)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"[TRT] ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()

    # Workspace
    if workspace_mib is None:
        workspace_mib = max(1024, get_max_workspace_bytes() >> 20)
    cap = int(os.getenv("TRT_WORKSPACE_CAP_MIB", "0"))  # optional cap
    if cap > 0:
        workspace_mib = min(workspace_mib, cap)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * (1 << 20))

    # Timing cache (speeds up repeated builds)
    try:
        tc_dir = timing_cache_dir or os.getenv("TRT_TIMING_CACHE_DIR") or tempfile.gettempdir()
        os.makedirs(tc_dir, exist_ok=True)
        tc_path = os.path.join(tc_dir, f"trt_{hashlib.sha1(os.path.abspath(onnx_path).encode()).hexdigest()[:10]}.blob")
        if hasattr(trt, "TimingCache"):
            try:
                with open(tc_path, "rb") as f:
                    cache = trt.TimingCache(f.read(), TRT_LOGGER)
            except Exception:
                cache = trt.TimingCache(TRT_LOGGER)
            config.set_timing_cache(cache, ignore_mismatch=True)
    except Exception:
        pass

    # Optimization profile
    in0 = network.get_input(0)
    H, W = default_hw
    # If network input already has static 4D shape, use that; else set min/opt/max.
    def _is_static_4d(shape):
        return (len(shape) == 4) and all(isinstance(x, int) and x > 0 for x in shape)
    shape = tuple(in0.shape)
    if _is_static_4d(shape):
        s_min = s_opt = s_max = shape
    else:
        s_min, s_opt, s_max = (1, 3, 320, 320), (1, 3, H, W), (1, 3, 1280, 1280)

    profile = builder.create_optimization_profile()
    profile.set_shape(in0.name, s_min, s_opt, s_max)
    config.add_optimization_profile(profile)

    # Precision
    if precision.lower() == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")

    # DLA + GPU fallback
    if device.startswith("dla") and hasattr(trt, "DeviceType"):
        config.default_device_type = trt.DeviceType.DLA
        try:
            core = int(device[-1]) if device in ("dla0", "dla1") else 0
            if hasattr(config, "DLA_core"):
                config.DLA_core = core
        except Exception:
            pass
        if hasattr(trt.BuilderFlag, "GPU_FALLBACK"):
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # Build
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("[TRT] Engine build returned None")

    # Save updated timing cache
    try:
        if hasattr(config, "get_timing_cache"):
            ser = config.get_timing_cache().serialize()
            with open(tc_path, "wb") as f:
                f.write(ser)
    except Exception:
        pass

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    return engine_path

# ---------- async wrapper (uses to_thread) ----------

async def optimize_onnx_to_engine(
    onnx_path: str,
    engine_path: str | None = None,
    *,
    precision: str = "fp16",
    device: str = "cuda",
    default_hw: tuple[int, int] = (640, 640),
    workspace_mib: int | None = None,
    reset_cuda: bool = False,
) -> str:
    """
    Async wrapper that offloads TRT engine build to a worker thread.
    """
    if engine_path is None:
        stem, _ = os.path.splitext(onnx_path)
        engine_path = stem + ".engine"

    if reset_cuda:
        # Drop any CUDA context created earlier by other libs/process parts
        maybe_reset_cuda_context()

    async with _BUILD_SEM:
        return await asyncio.to_thread(
            _trt_build_sync,
            onnx_path,
            engine_path,
            precision=precision,
            device=device,
            default_hw=default_hw,
            workspace_mib=workspace_mib,
            timing_cache_dir=os.getenv("TRT_TIMING_CACHE_DIR"),
        )
