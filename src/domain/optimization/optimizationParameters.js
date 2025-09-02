/**
 * Optimization 파라미터 그룹 정의
 * API 명세서에 따른 모든 최적화 파라미터들을 그룹별로 정리
 */

export const OPTIMIZATION_PARAM_GROUPS = {
  CONVERSION: 'Conversion Parameters',
  PRUNING: 'Pruning Parameters',
  OUTPUT: 'Output Parameters',
  GENERAL: 'General Parameters'
};

export const getOptimizationParameterGroups = (optimizationType) => {
  const baseGroups = [];

  // 최적화 타입별 특화 파라미터 추가 (API 명세서에 따라)
  if (optimizationType === 'pt_to_onnx_fp32' || optimizationType === 'pt_to_onnx_fp16') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.CONVERSION,
      params: [
        {
          key: 'input_size',
          label: 'Input Size [H, W]',
          type: 'array',
          default: [640, 640],
          desc: 'Input image size for ONNX conversion (Height, Width in pixels)',
          required: true
        },
        {
          key: 'batch_size',
          label: 'Batch Size',
          type: 'number',
          default: 1,
          min: 1,
          max: 32,
          step: 1,
          desc: 'Batch size for ONNX conversion'
        },
        {
          key: 'channels',
          label: 'Channels',
          type: 'number',
          default: 3,
          min: 1,
          max: 4,
          step: 1,
          desc: 'Number of input channels (RGB=3)'
        }
      ]
    });
  }

  if (optimizationType === 'onnx_to_trt') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.CONVERSION,
      params: [
        {
          key: 'precision',
          label: 'Precision',
          type: 'select',
          options: ['fp32', 'fp16'],
          default: 'fp32',
          desc: 'TensorRT precision mode (FP32 or FP16)'
        },
        {
          key: 'device',
          label: 'Device',
          type: 'select',
          options: ['gpu', 'dla', 'dla0', 'dla1'],
          default: 'gpu',
          desc: 'Target device (GPU or DLA cores for Jetson Orin)'
        }
      ]
    });
  }

  if (optimizationType === 'onnx_to_trt_int8') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.CONVERSION,
      params: [
        {
          key: 'calib_dir',
          label: 'Calibration Directory',
          type: 'text',
          default: '/app/int8_calib_images',
          desc: 'Container internal path to calibration images',
          placeholder: '/app/int8_calib_images'
        },
        {
          key: 'device',
          label: 'Device',
          type: 'select',
          options: ['gpu', 'dla'],
          default: 'gpu',
          desc: 'Target device (GPU or DLA for INT8)'
        },
        {
          key: 'mixed_fp16',
          label: 'Mixed FP16',
          type: 'checkbox',
          default: false,
          desc: 'Allow mixed FP16 operations for some layers'
        },
        {
          key: 'sparse',
          label: 'Sparse Weights',
          type: 'checkbox',
          default: false,
          desc: 'Enable sparse weights for INT8 quantization'
        },
        {
          key: 'int8_max_batches',
          label: 'Max Calibration Batches',
          type: 'number',
          default: 10,
          min: 1,
          max: 100,
          step: 1,
          desc: 'Maximum number of batches for calibration'
        },
        {
          key: 'input_size',
          label: 'Input Size [H, W]',
          type: 'array',
          default: [640, 640],
          desc: 'Input image size for INT8 calibration (Height, Width in pixels)',
          required: true
        },
        {
          key: 'workspace_mib',
          label: 'Workspace Size (MiB)',
          type: 'number',
          default: 2048,
          min: 512,
          max: 8192,
          step: 512,
          desc: 'TensorRT build workspace size in MiB'
        }
      ]
    });
  }

  if (optimizationType === 'prune_unstructured') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.PRUNING,
      params: [
        {
          key: 'amount',
          label: 'Pruning Amount',
          type: 'number',
          default: 0.2,
          min: 0.0,
          max: 0.9,
          step: 0.05,
          desc: 'Fraction of weights to prune (0.0 to 0.9)'
        },
        {
          key: 'pruning_type',
          label: 'Pruning Type',
          type: 'select',
          options: ['l1_unstructured', 'random_unstructured'],
          default: 'l1_unstructured',
          desc: 'Type of unstructured pruning to apply'
        }
      ]
    });
  }

  if (optimizationType === 'prune_structured') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.PRUNING,
      params: [
        {
          key: 'amount',
          label: 'Pruning Amount',
          type: 'number',
          default: 0.2,
          min: 0.0,
          max: 0.9,
          step: 0.05,
          desc: 'Fraction of channels to prune (0.0 to 0.9)'
        },
        {
          key: 'pruning_type',
          label: 'Pruning Type',
          type: 'select',
          options: ['ln_structured'],
          default: 'ln_structured',
          desc: 'Type of structured pruning to apply'
        },
        {
          key: 'n',
          label: 'L-norm Value',
          type: 'number',
          default: 2,
          min: 1,
          max: 10,
          step: 1,
          desc: 'L-norm value for structured pruning (L1, L2, etc.)'
        },
        {
          key: 'dim',
          label: 'Pruning Dimension',
          type: 'number',
          default: 0,
          min: 0,
          max: 3,
          step: 1,
          desc: 'Dimension to apply pruning (0 for output channels)'
        }
      ]
    });
  }

  // check_model_stats는 input_path만 필요하므로 추가 파라미터 없음

  return baseGroups;
};

// 파라미터 검증 함수
export const validateOptimizationParameter = (param, value) => {
  let error = '';
  
  if (param.type === 'number') {
    if (typeof value !== 'number' || isNaN(value)) {
      error = '숫자를 입력하세요.';
    } else if (param.min !== undefined && value < param.min) {
      error = `${param.label}은(는) 최소 ${param.min} 이상이어야 합니다.`;
    } else if (param.max !== undefined && value > param.max) {
      error = `${param.label}은(는) 최대 ${param.max} 이하여야 합니다.`;
    }
  } else if (param.type === 'text') {
    if (param.required && (!value || value === '')) {
      error = `${param.label}을(를) 입력하세요.`;
    }
  } else if (param.type === 'array') {
    if (param.required && (!Array.isArray(value) || value.length === 0)) {
      error = `${param.label}을(를) 입력하세요.`;
    }
  }
  
  return { isValid: error === '', error };
};

// 파라미터 값 정규화 함수
export const normalizeOptimizationParamValue = (value, param) => {
  if (param.type === 'number') {
    const numValue = Number(value);
    if (isNaN(numValue)) return param.default;
    
    // Apply min/max constraints
    if (param.min !== undefined && numValue < param.min) return param.min;
    if (param.max !== undefined && numValue > param.max) return param.max;
    
    // Apply step precision
    if (param.step !== undefined) {
      return Math.round(numValue / param.step) * param.step;
    }
    
    return numValue;
  }
  
  return value || param.default;
};
