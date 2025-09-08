import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const OPTIMIZATION_TYPES = [
  { 
    value: 'pt_to_onnx_fp32', 
    label: 'PT → ONNX FP32',
    supportedFormats: ['.pt'],
    description: 'PyTorch 모델을 ONNX FP32 형식으로 변환',
    inputRequirement: 'PyTorch .pt 파일만 지원'
  },
  { 
    value: 'pt_to_onnx_fp16', 
    label: 'PT → ONNX FP16',
    supportedFormats: ['.pt'],
    description: 'PyTorch 모델을 ONNX FP16 형식으로 변환',
    inputRequirement: 'PyTorch .pt 파일만 지원'
  },
  { 
    value: 'onnx_to_trt', 
    label: 'ONNX → TensorRT (FP32/FP16)',
    supportedFormats: ['.onnx'],
    description: 'ONNX 모델을 TensorRT FP32/FP16 형식으로 변환',
    inputRequirement: 'ONNX .onnx 파일만 지원'
  },
  { 
    value: 'onnx_to_trt_int8', 
    label: 'ONNX → TensorRT INT8',
    supportedFormats: ['.onnx'],
    description: 'ONNX 모델을 TensorRT INT8 형식으로 변환',
    inputRequirement: 'ONNX .onnx 파일만 지원'
  },
  { 
    value: 'prune_unstructured', 
    label: 'Unstructured Pruning',
    supportedFormats: ['.pt'],
    description: 'PyTorch 모델의 비구조적 가지치기',
    inputRequirement: 'PyTorch .pt 파일만 지원'
  },
  { 
    value: 'prune_structured', 
    label: 'Structured Pruning (Ln)',
    supportedFormats: ['.pt'],
    description: 'PyTorch 모델의 구조적 가지치기',
    inputRequirement: 'PyTorch .pt 파일만 지원'
  },
  { 
    value: 'check_model_stats', 
    label: 'Check Model Stats',
    supportedFormats: ['.pt', '.onnx', '.engine'],
    description: '모델 통계 정보 확인 (범용)',
    inputRequirement: 'PT/ONNX/TRT 모든 형식 지원'
  }
];

export default function OptimizationTypeSelector({ 
  optimizationType, 
  onOptimizationTypeChange, 
  disabled
}) {
  const selectedOptimization = OPTIMIZATION_TYPES.find(opt => opt.value === optimizationType);

  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Optimization Type</label>
      <select
        className={dsStyles.select}
        value={optimizationType || ''}
        onChange={e => onOptimizationTypeChange(e.target.value)}
        disabled={disabled}
      >
        <option value="">Select optimization type</option>
        {OPTIMIZATION_TYPES.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      
    </div>
  );
}
