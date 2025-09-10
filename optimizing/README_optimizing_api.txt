got it—here’s a **drop-in README** update that adds:

* **GPU fallback behavior** when DLA is used
* **Jetson Orin has 2 DLA cores (dla0, dla1)** and how to target them
* A small **concurrency / resource management** section tied to your `_BUILD_SEM` and TRT fallback
* Keeps the earlier option-by-option details and the **\[H, W]** explanation

---

# Optimizing API 사용 설명서

이 문서는 `/optimizing` 컨테이너가 제공하는 모든 모델 최적화 엔드포인트의 **요청 형식, 필드 옵션, 기본값, 제약**을 정리합니다.

---

## 설계 요약

* **입력/출력 경로 명확화**: 입력은 보통 **MinIO key**를 사용하고, 출력은 파일명을 주면 **작업 디렉토리(OID) 아래**로 정리됩니다.
* **자동 산출물**: 모든 작업 완료 시 모델과 함께 `<stem>_stats.txt`가 생성되어 MinIO(`artifacts/{pid}/optimizing/{oid}/...`)로 업로드됩니다.
* **메트릭 저장**: MongoDB에는 결과와 함께 상세 `metrics`가 저장됩니다.
* **간단한 요청**: API 경로가 작업을 결정하므로 일반적으로 **`pid` + `parameters`** 만 보내면 됩니다.
  (`uid/oid`는 인증/라우터에서 채워지고, `action`은 라우트로 결정)

---

## 공통 규칙

### 인증/식별자

* `uid`/`oid`는 서버가 채웁니다(라우트/미들웨어).
* 클라이언트는 **`pid`** 와 **`parameters`** 만 보내면 됩니다.

### 경로 처리

* `parameters.input_path`

  * **권장:** MinIO key (예: `artifacts/P0001/optimizing/O0001/yolov8n.pt`)
  * **허용:** 컨테이너 내부 **절대경로**(디버깅용). 절대경로이면 그대로 사용.
* `parameters.output_path`

  * **파일명만** 주는 것을 권장 → **작업 디렉토리**에 저장.
  * **디렉토리 포함 경로를 줘도 파일명만 사용**(서브폴더는 무시).
  * 일부 작업은 **생략 가능**(자동 파일명 생성), 일부는 **필수**(아래 각 섹션 참고).

### 결과 업로드

* 작업 종료 시 **디렉토리/파일**이 MinIO 경로 `artifacts/{pid}/optimizing/{oid}/...`로 업로드됩니다.
* 결과 JSON에는 `artifacts_path`, `metrics`, `artifact_files` 등이 포함됩니다.

### \[H, W] 표기 (중요)

* **`[H, W]` = \[Height, Width] (픽셀 단위)**
  예: `[640, 640]` → 높이 640px × 너비 640px.
  내부 텐서 순서는 보통 **NCHW**(배치 N, 채널 C, 높이 H, 너비 W).
  모델이 기대하는 입력 크기와 동일하게 설정해야 성능/정확도 하락 방지.

---

## 응답 형식 (공통)

* HTTP 202, 본문 예:

  ```json
  {
    "uid": "0001",
    "pid": "P0001",
    "oid": "O0007",
    "action": "onnx_to_trt",
    "status": "started",
    "details": { "kind": "onnx_to_trt" },
    "info": { "workdir": "/workspace/shared/jobs/0001/P0001/O0007" }
  }
  ```
* 백그라운드 완료 후 콜백/DB에 **최종 상태(`completed`/`failed`) + metrics** 반영.

---

## 엔드포인트별 사양

### 1) POST `/optimizing/pt_to_onnx_fp32`

**기능:** `.pt` → **ONNX FP32** 내보내기

**parameters**

* `kind` *(필수)*: `"pt_to_onnx"`
* `input_path` *(필수)*: MinIO key 또는 절대경로
* `output_path` *(선택)*: 결과 파일명 (예: `yolov8n_fp32.onnx`)
  → 생략 시 자동: `<stem>_fp32.onnx` (작업 디렉토리 내부)
* `input_size` *(선택, 기본 `[640, 640]`)*: `[H, W]` (픽셀)
* `batch_size` *(선택, 기본 `1`)*
* `channels` *(선택, 기본 `3`)*: 입력 채널 수(RGB=3)

**예시**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "pt_to_onnx",
    "input_path": "artifacts/P0001/optimizing/O0001/best.pt",
    "input_size": [640, 640],
    "batch_size": 1
  }
}
```

---

### 2) POST `/optimizing/pt_to_onnx_fp16`

**기능:** `.pt` → **ONNX FP16** 내보내기

**parameters** *(FP32와 동일, precision만 FP16)*

* `kind` *(필수)*: `"pt_to_onnx"`
* `input_path` *(필수)*
* `output_path` *(선택)*: 생략 시 자동 `<stem>_fp16.onnx`
* `input_size` *(선택, 기본 `[640, 640]`)*
* `batch_size` *(선택, 기본 `1`)*
* `channels` *(선택, 기본 `3`)*

**예시**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "pt_to_onnx",
    "input_path": "artifacts/P0001/optimizing/O0001/yolov8n.pt",
    "output_path": "yolov8n_fp16.onnx",
    "input_size": [640, 640],
    "batch_size": 1
  }
}
```

---

### 3) POST `/optimizing/prune_unstructured`

**기능:** **비구조적**(weight 개별) 프루닝

**parameters**

* `kind` *(필수)*: `"prune_unstructured"`
* `input_path` *(필수)*
* `output_path` *(필수)*: 결과 `.pt` 파일명
* `amount` *(선택, 기본 `0.2`)*: `0.0 ~ 1.0` (제거 비율)
* `pruning_type` *(선택, 기본 `"l1_unstructured"`)*:
  허용 `"l1_unstructured"`, `"random_unstructured"`

**예시**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "prune_unstructured",
    "input_path": "artifacts/P0001/optimizing/O0001/best.pt",
    "amount": 0.2,
    "pruning_type": "l1_unstructured"
  }
}
```

---

### 4) POST `/optimizing/prune_structured`

**기능:** **구조적**(채널/필터 단위) 프루닝 (Ln)

**parameters**

* `kind` *(필수)*: `"prune_structured"`
* `input_path` *(필수)*
* `output_path` *(필수)*
* `amount` *(선택, 기본 `0.2`)*: `0.0 ~ 1.0`
* `pruning_type` *(선택, 기본 `"ln_structured"`)*
* `n` *(선택, 기본 `2`)*: L-norm (보통 1 또는 2)
* `dim` *(선택, 기본 `0`)*: 적용 차원 (Conv2d에서 보통 `0`/`1`)

**예시**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "prune_structured",
    "input_path": "artifacts/P0001/optimizing/O0001/best.pt",
    "amount": 0.2,
    "pruning_type": "ln_structured",
    "n": 2,
    "dim": 0
  }
}
```

---

### 5) POST `/optimizing/check_model_stats`

**기능:** 모델 통계 산출(파일 크기, 파라미터 수, 희소도 등) → `<stem>_stats.txt`

**parameters**

* `kind` *(필수)*: `"check_model_stats"`
* `input_path` *(필수)*

**예시**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "check_model_stats",
    "input_path": "artifacts/P0001/optimizing/O0001/best.pt"
  }
}
```

---

### 6) POST `/optimizing/onnx_to_trt`

**기능:** **ONNX → TensorRT 엔진** 빌드 (**FP32/FP16**, trtexec 사용)

**parameters**

* `kind` *(필수)*: `"onnx_to_trt"`
* `input_path` *(필수)*: ONNX 경로/MinIO key
* `output_path` **(필수)**: 결과 **엔진 파일명** (예: `yolov8n_fp16.engine`)
* `precision` *(필수)*: `"fp32"` 또는 `"fp16"`
* `device` *(선택, 기본 `"gpu"`)*:

  * `"gpu"`: GPU 엔진
  * `"dla"`, `"dla0"`, `"dla1"`: **Jetson Orin의 DLA 코어 사용**
    (Orin에는 **2개의 DLA 코어**가 있음: `dla0`, `dla1`. `"dla"`는 코어 **0**에 매핑)

**예시 (GPU FP16)**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "onnx_to_trt",
    "input_path": "artifacts/P0001/optimizing/O0014/best.onnx",
    "precision": "fp16",
    "device": "gpu"
  }
}
```

**예시 (DLA FP16, 코어 지정)**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "onnx_to_trt",
    "input_path": "artifacts/P0001/optimizing/O0014/best.onnx",
    "precision": "fp16",
    "device": "dla0"
  }
}
```

**DLA / GPU Fallback 주의사항**

* TensorRT는 DLA에서 **지원되지 않는 연산**이 있거나, **DLA 코어가 모두 점유 중**일 때
  **자동으로 GPU fallback**(GPU 실행)하도록 구성되어 있습니다(`--allowGPUFallback`).
* Jetson Orin의 DLA에는 **차원 제한(각 텐서 축 최대 8192)** 이 있습니다.
  예: YOLOv8 640×640은 출력 축 중 하나가 **8400** → 일부 레이어가 DLA에 못 올라가 **GPU fallback**.
* DLA 활용도를 높이려면 입력을 **\[608, 608]** 또는 **\[576, 576]** 등으로 낮추는 것을 권장
  (예: 608 → 76²+38²+19²=**7581** ≤ 8192)

**동적/정적 shape**

* ONNX가 **static shape**면 trtexec의 shape 플래그는 자동 생략.
* ONNX가 **dynamic shape**면 `min/opt/max`를 자동 설정합니다.
  (기본 `opt` 크기는 `TRT_DEFAULT_H`, `TRT_DEFAULT_W` 환경변수로 조정 가능)

---

### 7) POST `/optimizing/onnx_to_trt_int8`

**기능:** **ONNX → TensorRT INT8 양자화 엔진** (내장 엔트로피 캘리브레이터)

**parameters**

* `kind` *(필수)*: `"onnx_to_trt_int8"`
* `input_path` *(필수)*
* `output_path` **(필수)**: 엔진 파일명 (예: `yolov8n_int8.engine`)
* `calib_dir` **(필수)**: **컨테이너 내부** 이미지 폴더 (필터: `*.jpg|*.jpeg|*.png|*.bmp`)
* `precision` *(필수, 고정)*: `"int8"`
* `device` *(선택, 기본 `"gpu"`)*: `"gpu"` 또는 `"dla"`
  **주의:** INT8 스키마는 현재 `"dla0"`/`"dla1"`를 허용하지 않음 → `"dla"`는 **코어 0** 사용
* `mixed_fp16` *(선택, 기본 `false`)*: 일부 연산 FP16 혼합 허용
* `sparse` *(선택, 기본 `false`)*: Sparse weights 사용
* `int8_max_batches` *(선택, 기본 `10`)*: 캘리브레이션 배치 수 상한
* `input_size` *(필수)*: `[H, W]` (픽셀)
* `workspace_mib` *(선택, 기본 `2048`)*: 빌드 워크스페이스(MiB)

**예시 (DLA INT8)**

```json
{
  "pid": "P0001",
  "parameters": {
    "kind": "onnx_to_trt_int8",
    "input_path": "artifacts/P0001/optimizing/O0014/best.onnx",
    "calib_dir": "/app/int8_calib_images",
    "device": "dla",
    "mixed_fp16": false,
    "sparse": false,
    "precision": "int8",
    "int8_max_batches": 10,
    "input_size": [608, 608]
  }
}
```

**DLA / GPU Fallback 주의사항**

* DLA 미지원 연산 또는 DLA 코어 점유 시 **자동 GPU fallback**.
* DLA의 **8192 차원 제한** 고려(입력을 608/576 등으로 낮추면 더 많은 레이어가 DLA에서 실행).

---

## 동시성 / 자원 관리

* 빌드 동시성: `ENGINE_BUILD_CONCURRENCY` (기본 **3**)
  내부적으로 `asyncio.Semaphore` 로 빌드 태스크 동시 실행 수를 제한합니다.
* **Jetson Orin은 DLA 코어가 2개(dla0, dla1)** 이므로, DLA 대상 작업을 **2개 초과** 동시에 던지면
  세 번째 이후 작업은 TensorRT의 **GPU fallback**이 더 많이 발생할 수 있습니다.

  * 확실히 DLA만 사용하고 싶다면:

    * 동시성을 **2 이하**로 설정하거나,
    * 여분 작업은 `device: "gpu"` 로 보내 DLA 혼잡을 피하세요.
* 추가 튜닝

  * `TRT_WORKSPACE_MIB`(trtexec) 또는 INT8의 `workspace_mib` 로 워크스페이스 크기 조정
  * `TRTEXEC_TIMEOUT_SEC` 으로 trtexec 빌드 타임아웃 지정
  * **Timing Cache** 공유로 빌드 시간 단축: `TRT_TIMING_CACHE_DIR=/workspace/shared/timing_cache`

---

## 산출물 & 메트릭

* **공통**

  * `artifact_files`: 생성 파일 목록(예: `["yolov8n_fp16.engine", "yolov8n_fp16_stats.txt"]`)
  * `stats`: 크기(MB), total/nonzero 파라미터, sparsity(%)
* **TRT 엔진(`.engine`) 추가 정보(가능 시)**

  * `tensorrt_version`, `device_memory_mb`
  * IO 텐서(name/dtype/shape) 요약

---

## 오류/제약 요약

* `/onnx_to_trt`, `/onnx_to_trt_int8`는 **`output_path` 필수**(파일명만 전달 권장).
* `/onnx_to_trt_int8`는 **`calib_dir` 필수**이며 **컨테이너 내부 경로**여야 함.
* DLA는 해당 하드웨어(Jetson Xavier/Orin 등)에서만 유효.
  Orin에는 **DLA 2개(dla0, dla1)** 가 있으며, 부족 시 **GPU fallback** 발생 가능.
* INT8 스키마의 `device`는 `"gpu"`/`"dla"`만 허용 (`"dla0"`/`"dla1"`은 현 시점 미지원 → 기본 코어 0 사용).

---

필요 시 INT8에서 `dla0`/`dla1`도 선택 가능하게 **Pydantic 스키마(OnnxToTrtInt8Params.device)** 를 `Literal["gpu","dla","dla0","dla1"]` 로 확장하고, 라우터 입력 검증만 조정하면 바로 동작합니다.
