/**
 * Optimization 파라미터 그룹 정의
 * API 설명서에 따른 모든 최적화 파라미터들을 그룹별로 정리
 */

export const OPTIMIZATION_PARAM_GROUPS = {
  CONVERSION: 'Conversion Parameters',
  PRUNING: 'Pruning Parameters',
  OUTPUT: 'Output Parameters',
  GENERAL: 'General Parameters'
};

export const getOptimizationParameterGroups = (optimizationType) => {
  const baseGroups = [
    {
      group: OPTIMIZATION_PARAM_GROUPS.GENERAL,
      params: [
        {
          key: 'training_id',
          label: 'Training ID',
          type: 'text',
          required: true,
          desc: 'Training ID to use for optimization (e.g., T0001)',
          placeholder: 'T0001'
        },
        {
          key: 'model_name',
          label: 'Model Name',
          type: 'text',
          required: true,
          desc: 'Model file name (e.g., yolov8n.pt)',
          placeholder: 'yolov8n.pt'
        }
      ]
    }
  ];

  // 최적화 타입별 특화 파라미터 추가
  if (optimizationType === 'pt_to_onnx_fp32' || optimizationType === 'pt_to_onnx_fp16') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.CONVERSION,
      params: [
        {
          key: 'input_size',
          label: 'Input Size',
          type: 'array',
          default: [640, 640],
          desc: 'Input image size for ONNX conversion (width, height)',
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
          key: 'opset_version',
          label: 'ONNX Opset Version',
          type: 'number',
          default: 11,
          min: 7,
          max: 17,
          step: 1,
          desc: 'ONNX opset version for conversion'
        },
        {
          key: 'dynamic_axes',
          label: 'Dynamic Axes',
          type: 'checkbox',
          default: false,
          desc: 'Enable dynamic axes for variable input sizes'
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
        },
        {
          key: 'global_unstructured',
          label: 'Global Pruning',
          type: 'checkbox',
          default: false,
          desc: 'Apply pruning globally across all layers'
        },
        {
          key: 'importance',
          label: 'Importance Function',
          type: 'select',
          options: ['magnitude', 'random', 'gradient'],
          default: 'magnitude',
          desc: 'Function to determine weight importance'
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

  // check_model_stats에만 통계 관련 파라미터 추가
  if (optimizationType === 'check_model_stats') {
    baseGroups.push({
      group: OPTIMIZATION_PARAM_GROUPS.OUTPUT,
      params: [
        {
          key: 'save_stats',
          label: 'Save Statistics',
          type: 'checkbox',
          default: true,
          desc: 'Generate and save model statistics report'
        },
        {
          key: 'detailed_stats',
          label: 'Detailed Statistics',
          type: 'checkbox',
          default: false,
          desc: 'Include detailed layer-wise statistics'
        },
        {
          key: 'save_format',
          label: 'Save Format',
          type: 'select',
          options: ['json', 'txt', 'csv'],
          default: 'json',
          desc: 'Format for saving statistics'
        }
      ]
    });
  }

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
