import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const OPTIMIZATION_TYPES = [
  { value: 'pt_to_onnx_fp32', label: 'PT → ONNX FP32' },
  { value: 'pt_to_onnx_fp16', label: 'PT → ONNX FP16' },
  { value: 'onnx_to_trt', label: 'ONNX → TensorRT (FP32/FP16)' },
  { value: 'onnx_to_trt_int8', label: 'ONNX → TensorRT INT8' },
  { value: 'prune_unstructured', label: 'Unstructured Pruning' },
  { value: 'prune_structured', label: 'Structured Pruning (Ln)' },
  { value: 'check_model_stats', label: 'Check Model Stats' }
];

export default function OptimizationTypeSelector({ 
  optimizationType, 
  onOptimizationTypeChange, 
  disabled
}) {
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
