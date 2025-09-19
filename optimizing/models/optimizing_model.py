# API-only input model for request body (no uid/action required)
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# ===== Enums =====
class Action(str, Enum):
    pt_to_onnx = "pt_to_onnx"
    prune_unstructured = "prune_unstructured"
    prune_structured = "prune_structured"
    onnx_to_trt = "onnx_to_trt"
    onnx_to_trt_int8 = "onnx_to_trt_int8"
    check_model_stats = "check_model_stats"

class Status(str, Enum):
    started = "started"
    completed = "completed"
    failed = "failed"

# ===== Info =====
class OptimizingInfo(BaseModel):
    uid: str
    pid: str
    oid: Optional[str] = None
    action: Action
    workdir: str

# ===== Params =====
class PruneUnstructuredParams(BaseModel):
    kind: Literal["prune_unstructured"] = "prune_unstructured"
    input_path: str
    output_path: Optional[str] = None  # Output will always be named 'best.pt'
    amount: float = 0.2
    pruning_type: Literal["l1_unstructured", "random_unstructured"] = "l1_unstructured"
    info: Optional[OptimizingInfo] = None

class PruneStructuredParams(BaseModel):
    kind: Literal["prune_structured"] = "prune_structured"
    input_path: str
    output_path: Optional[str] = None  # Output will always be named 'best.pt'
    amount: float = 0.2
    pruning_type: Literal["ln_structured"] = "ln_structured"
    dim: int = 0
    n: int = 2
    info: Optional[OptimizingInfo] = None

class PtToOnnxParams(BaseModel):
    kind: Literal["pt_to_onnx"] = "pt_to_onnx"
    input_path: str
    output_path: Optional[str] = None  # Output will always be named 'best.onnx'
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 1
    info: Optional[OptimizingInfo] = None

class OnnxToTrtParams(BaseModel):
    kind: Literal["onnx_to_trt"] = "onnx_to_trt"
    input_path: str
    output_path: Optional[str] = None  # Output will always be named 'best.engine'
    device: str = "gpu"  # allow "gpu", "dla", "dla0", "dla1", ...
    precision: Literal["fp32", "fp16"] = "fp16"
    info: Optional[OptimizingInfo] = None

class OnnxToTrtInt8Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["onnx_to_trt_int8"] = "onnx_to_trt_int8"
    input_path: str
    output_path: Optional[str] = None  # Output will always be named 'best.engine'
    calib_dir: str
    calibrator: Literal["entropy"] = "entropy"
    device: str = "gpu"          # allow "gpu", "dla", "dla0", "dla1", ...
    mixed_fp16: bool = False
    sparse: bool = False
    precision: Literal["int8"] = "int8"
    int8_max_batches: int = 10
    input_size: Tuple[int, int] = (640, 640)
    workspace_mib: int = 2048
    info: Optional[OptimizingInfo] = None

class CheckModelStatsParams(BaseModel):
    kind: Literal["check_model_stats"] = "check_model_stats"
    input_path: str
    info: Optional[OptimizingInfo] = None

OptimizingParameters = Union[
    PruneUnstructuredParams,
    PruneStructuredParams,
    PtToOnnxParams,
    OnnxToTrtParams,
    OnnxToTrtInt8Params,
    CheckModelStatsParams,
]

# ===== Request/Result =====
class OptimizingRequest(BaseModel):
    uid: str
    pid: str
    oid: Optional[str] = None
    action: Action
    parameters: OptimizingParameters = Field(..., discriminator="kind")
    info: Optional[OptimizingInfo] = None

class OptimizingResult(BaseModel):
    uid: str
    pid: str
    oid: Optional[str] = None
    action: Action | str
    status: Status | str
    message: Optional[str] = None
    artifacts_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    info: Optional[OptimizingInfo] = None

class OptimizingHandlingRequest(BaseModel):
    workdir: str
    request: OptimizingRequest
    info: Optional[OptimizingInfo] = None

class OptimizingRequestBody(BaseModel):
    pid: str
    oid: Optional[str] = None
    parameters: OptimizingParameters
    info: Optional[dict] = None