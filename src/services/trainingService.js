// 파라미터 검증 유틸리티
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

// step에서 소수점 자릿수 계산
export const getDecimalPlaces = (step) => {
  if (!step || step >= 1) return 0;
  return step.toString().split('.')[1]?.length || 0;
};

// 파라미터 값 정규화
export const normalizeParamValue = (value, param) => {
  if (param && param.type === 'number') {
    const decimals = getDecimalPlaces(param.step);
    return Number(Number(value).toFixed(decimals));
  }
  return value;
};

// Training 실행 검증
export const validateTrainingExecution = (trainingType, selectedDataset, selectedSnapshot, mode) => {
  const errors = [];
  
  if (mode === 'no-code') {
    if (!selectedDataset) {
      errors.push('Please select a dataset.');
    }
    if (!selectedSnapshot && trainingType === 'standard') {
      errors.push('Please select a snapshot.');
    }
  }
  
  if (trainingType === 'continual' && !selectedSnapshot) {
    errors.push('Base snapshot is required for continual training.');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Training 실행
export const executeTraining = async (trainingConfig) => {
  const { trainingType, selectedDataset, selectedSnapshot, algorithm, algoParams } = trainingConfig;
  
  // 실제 API 호출 대신 mock 실행
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        success: true,
        message: 'Training started successfully',
        trainingId: `train_${Date.now()}`
      });
    }, 1000);
  });
};

// 파라미터 그룹 가져오기
export const getParameterGroups = (algorithm) => {
  const yolov8ParamGroups = [
    {
      group: 'Project Information',
      params: [
        { key: 'model_version', label: 'Model Version', type: 'select', options: ['v8', 'v7', 'v6'], default: 'v8', required: true },
        { key: 'model_size', label: 'Model Size', type: 'select', options: ['n', 's', 'm', 'l', 'x'], default: 'n', required: true },
        { key: 'task_type', label: 'Task Type', type: 'select', options: ['object_detection', 'classification', 'segmentation'], default: 'object_detection', required: true },
      ],
    },
    {
      group: 'Data Configuration',
      params: [
        { key: 'split_ratio', label: 'Train/Val Split Ratio', type: 'number', min: 0.1, max: 0.9, step: 0.05, default: 0.8 },
        { key: 'dataset_path', label: 'Dataset Path', type: 'text', default: '' },
      ],
    },
    {
      group: 'Training Parameters',
      params: [
        { key: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1, default: 100 },
        { key: 'batch', label: 'Batch Size', type: 'number', min: 1, max: 1024, step: 1, default: 16 },
        { key: 'imgsz', label: 'Image Size', type: 'number', min: 32, max: 4096, step: 32, default: 640 },
        { key: 'lr0', label: 'Initial Learning Rate', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'lrf', label: 'Final Learning Rate Fraction', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'momentum', label: 'Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.937 },
        { key: 'weight_decay', label: 'Weight Decay', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.0005 },
        { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['SGD', 'Adam', 'AdamW'], default: 'SGD' },
      ],
    },
    {
      group: 'Advanced Training',
      params: [
        { key: 'warmup_epochs', label: 'Warmup Epochs', type: 'number', min: 0, max: 100, step: 1, default: 3 },
        { key: 'warmup_momentum', label: 'Warmup Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.8 },
        { key: 'warmup_bias_lr', label: 'Warmup Bias LR', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.1 },
        { key: 'box', label: 'Box Loss Gain', type: 'number', min: 0, max: 10, step: 0.1, default: 0.05 },
        { key: 'cls', label: 'Class Loss Gain', type: 'number', min: 0, max: 10, step: 0.1, default: 0.5 },
        { key: 'dfl', label: 'DFL Loss Gain', type: 'number', min: 0, max: 10, step: 0.1, default: 1.5 },
        { key: 'fl_gamma', label: 'Focal Loss Gamma', type: 'number', min: 0, max: 10, step: 0.1, default: 0.0 },
        { key: 'label_smoothing', label: 'Label Smoothing', type: 'number', min: 0, max: 1, step: 0.01, default: 0.0 },
        { key: 'nbs', label: 'Nominal Batch Size', type: 'number', min: 1, max: 1024, step: 1, default: 64 },
        { key: 'dropout', label: 'Dropout Rate', type: 'number', min: 0, max: 1, step: 0.01, default: 0.0 },
      ],
    },
  ];

  const algorithm1Params = [
    { key: 'paramA', label: 'Param A', type: 'number', min: 0, max: 100, step: 1, default: 10 },
    { key: 'paramB', label: 'Param B', type: 'text', default: '' },
  ];

  const algorithm2Params = [
    { key: 'alpha', label: 'Alpha', type: 'number', min: 0, max: 1, step: 0.01, default: 0.5 },
  ];

  switch (algorithm) {
    case 'YOLO':
      return yolov8ParamGroups;
    case 'Faster R-CNN':
      return [{ group: 'Faster R-CNN Parameters', params: algorithm1Params }];
    case 'SSD':
      return [{ group: 'SSD Parameters', params: algorithm2Params }];
    default:
      return [];
  }
};

// 알고리즘 옵션
export const algorithmOptions = [
  { value: 'YOLO', label: 'YOLO' },
  { value: 'Faster R-CNN', label: 'Faster R-CNN' },
  { value: 'SSD', label: 'SSD' },
]; 