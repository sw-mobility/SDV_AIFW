// Parameter Groups Domain Model
export const PARAMETER_GROUPS = {
  PROJECT_INFO: 'Project Information',
  TRAINING: 'Training Parameters',
  MODEL: 'Model Parameters',
  DATA: 'Data Parameters'
};

export const PROJECT_INFO_PARAMS = ['model_version', 'model_size', 'task_type'];

// Algorithm options for selector
export const algorithmOptions = [
  { value: 'YOLO', label: 'YOLO' },
  { value: 'ResNet', label: 'ResNet' },
  { value: 'SSD', label: 'SSD' },
  { value: 'Faster R-CNN', label: 'Faster R-CNN' }
];

// Parameter validation and utility functions
export const validateParam = (param, value) => {
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
  }
  
  return { isValid: error === '', error };
};

export const normalizeParamValue = (value, param) => {
  if (param.type === 'number') {
    const numValue = Number(value);
    if (isNaN(numValue)) return param.default;
    
    // Apply min/max constraints
    if (param.min !== undefined && numValue < param.min) return param.min;
    if (param.max !== undefined && numValue > param.max) return param.max;
    
    // Apply step precision
    if (param.step !== undefined) {
      const step = param.step;
      const precision = getDecimalPlaces(step);
      return Math.round(numValue / step) * step;
    }
    
    return numValue;
  }
  
  return value;
};

export const getDecimalPlaces = (num) => {
  if (Math.floor(num) === num) return 0;
  return num.toString().split('.')[1].length || 0;
};

export const getParameterGroupsByAlgorithm = (algorithm) => {
  // 기본 파라미터 그룹 정의
  const baseGroups = [
    {
      group: PARAMETER_GROUPS.PROJECT_INFO,
      params: [
        { key: 'model_version', label: 'Model Version', type: 'text', required: true },
        { key: 'model_size', label: 'Model Size', type: 'text', required: true },
        { key: 'task_type', label: 'Task Type', type: 'text', required: true },
        { key: 'project_name', label: 'Project Name', type: 'text', required: true },
        { key: 'description', label: 'Description', type: 'text' }
      ]
    },
    {
      group: PARAMETER_GROUPS.TRAINING,
      params: [
        { key: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, default: 100, step: 1 },
        { key: 'batch_size', label: 'Batch Size', type: 'number', min: 1, max: 512, default: 32, step: 1 },
        { key: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.0001, max: 1, default: 0.001, step: 0.0001 },
        { key: 'optimizer', label: 'Optimizer', type: 'text', default: 'adam' }
      ]
    },
    {
      group: PARAMETER_GROUPS.MODEL,
      params: [
        { key: 'model_type', label: 'Model Type', type: 'text', default: 'yolo' },
        { key: 'input_size', label: 'Input Size', type: 'number', min: 224, max: 1024, default: 640, step: 32 },
        { key: 'num_classes', label: 'Number of Classes', type: 'number', min: 1, max: 1000, default: 80, step: 1 }
      ]
    },
    {
      group: PARAMETER_GROUPS.DATA,
      params: [
        { key: 'data_path', label: 'Data Path', type: 'text', required: true },
        { key: 'validation_split', label: 'Validation Split', type: 'number', min: 0.1, max: 0.5, default: 0.2, step: 0.05 },
        { key: 'augmentation', label: 'Data Augmentation', type: 'text', default: 'basic' }
      ]
    }
  ];

  // 알고리즘별 특화 파라미터 추가
  if (algorithm === 'YOLO') {
    baseGroups[2].params.push(
      { key: 'yolo_version', label: 'YOLO Version', type: 'text', default: 'v8', desc: 'YOLO version to use' },
      { key: 'confidence_threshold', label: 'Confidence Threshold', type: 'number', min: 0.1, max: 1, default: 0.5, step: 0.01 }
    );
  } else if (algorithm === 'ResNet') {
    baseGroups[2].params.push(
      { key: 'resnet_depth', label: 'ResNet Depth', type: 'number', min: 18, max: 152, default: 50, step: 1 },
      { key: 'pretrained', label: 'Use Pretrained', type: 'text', default: 'true' }
    );
  }

  return baseGroups;
}; 