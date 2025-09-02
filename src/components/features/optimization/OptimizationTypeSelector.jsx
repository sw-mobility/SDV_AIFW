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
      
      {!optimizationType && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          backgroundColor: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: '6px',
          fontSize: '14px',
          color: '#64748b',
          textAlign: 'center'
        }}>
          최적화 타입을 선택하면 파라미터 설정과 실행 옵션이 표시됩니다.
        </div>
      )}
    </div>
  );
}
