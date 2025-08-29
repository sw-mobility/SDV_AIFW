// Parameter Groups Domain Model
export const PARAMETER_GROUPS = {
  PROJECT_INFO: 'Project Information',
  TRAINING: 'Training Parameters',
  MODEL: 'Model Parameters',
  DATA: 'Data Parameters',
  PREPROCESSING: 'Preprocessing',
  ADVANCED: 'Advanced Parameters'
};

export const PROJECT_INFO_PARAMS = ['model_version', 'model_size', 'task_type'];

// Algorithm options for selector
export const algorithmOptions = [
  { value: 'yolo_v5', label: 'YOLOv5' },
  { value: 'yolo_v8', label: 'YOLOv8' },
  { value: 'yolo_v11', label: 'YOLOv11' },
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
  } else if (param.type === 'select') {
    if (param.required && (!value || value === '')) {
      error = `${param.label}을(를) 선택하세요.`;
    }
  } else if (param.type === 'checkbox') {
    // checkbox는 boolean 값이므로 별도 검증 불필요
  } else if (param.type === 'yaml_editor') {
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
  } else if (param.type === 'yaml_editor') {
    // YAML 에디터는 문자열 값 그대로 반환
    return value || param.default || '';
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
    // Project Information 파라미터들은 실제 API에 필요하지 않으므로 제거
    // {
    //   group: PARAMETER_GROUPS.PROJECT_INFO,
    //   params: [
    //     { key: 'model_version', label: 'Model Version', type: 'text', required: true, default: 'v1.0' },
    //     { key: 'model_size', label: 'Model Size', type: 'text', required: true, default: 'medium' },
    //     { key: 'task_type', label: 'Task Type', type: 'text', required: true, default: 'training' },
    //     { key: 'project_name', label: 'Project Name', type: 'text', required: true },
    //     { key: 'description', label: 'Description', type: 'text' }
    //   ]
    // },
    {
      group: PARAMETER_GROUPS.PREPROCESSING,
      params: [
        { key: 'coco_classes', label: 'COCO Classes', type: 'yaml_editor', default: '', desc: 'Customize COCO dataset classes' }
      ]
    },
    {
      group: PARAMETER_GROUPS.TRAINING,
      params: [
        { key: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, default: 50, step: 1, desc: 'Number of training epochs' },
        { key: 'batch_size', label: 'Batch Size', type: 'number', min: 1, max: 512, default: 16, step: 1, desc: 'Batch size for training' },
        { key: 'learning_rate', label: 'Learning Rate (lr0)', type: 'number', min: 0.0001, max: 1, default: 0.01, step: 0.001, desc: 'Initial learning rate' },
        { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['SGD', 'Adam', 'AdamW'], default: 'SGD', desc: 'Optimizer algorithm' }
      ]
    },
    {
      group: PARAMETER_GROUPS.MODEL,
      params: [
        { key: 'model_type', label: 'Model Type', type: 'text', default: 'yolo', desc: 'Model architecture type' },
        { key: 'input_size', label: 'Input Size (imgsz)', type: 'number', min: 224, max: 1024, default: 640, step: 32, desc: 'Input image size' },
        { key: 'num_classes', label: 'Number of Classes', type: 'number', min: 1, max: 1000, default: 80, step: 1, desc: 'Number of classes to detect' }
      ]
    },
    {
      group: PARAMETER_GROUPS.DATA,
      params: [
        { key: 'data_path', label: 'Data Path', type: 'text', required: true, desc: 'Path to dataset' },
        { key: 'validation_split', label: 'Validation Split', type: 'number', min: 0.1, max: 0.5, default: 0.2, step: 0.05, desc: 'Validation data ratio' },
        { key: 'augmentation', label: 'Data Augmentation', type: 'checkbox', default: true, desc: 'Enable data augmentation' }
      ]
    }
  ];

  // YOLO 특화 파라미터 추가
  if (algorithm === 'YOLO') {
    baseGroups[1].params.push(
      { key: 'lrf', label: 'Final LR Factor', type: 'number', min: 0.01, max: 1, default: 0.1, step: 0.01, desc: 'Final learning rate factor' },
      { key: 'momentum', label: 'Momentum', type: 'number', min: 0, max: 1, default: 0.937, step: 0.001, desc: 'SGD momentum/Adam beta1' },
      { key: 'weight_decay', label: 'Weight Decay', type: 'number', min: 0, max: 0.01, default: 0.0005, step: 0.0001, desc: 'Optimizer weight decay' },
      { key: 'patience', label: 'Patience', type: 'number', min: 1, max: 100, default: 20, step: 1, desc: 'Early stopping patience' },
      { key: 'warmup_epochs', label: 'Warmup Epochs', type: 'number', min: 0, max: 10, default: 3, step: 1, desc: 'Warmup epochs' },
      { key: 'warmup_momentum', label: 'Warmup Momentum', type: 'number', min: 0, max: 1, default: 0.8, step: 0.1, desc: 'Warmup initial momentum' },
      { key: 'warmup_bias_lr', label: 'Warmup Bias LR', type: 'number', min: 0, max: 1, default: 0.1, step: 0.01, desc: 'Warmup initial bias lr' }
    );
    
    baseGroups[2].params.push(
      { key: 'model', label: 'Model', type: 'select', options: ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], default: 'yolov8n', desc: 'YOLO model variant' },
      { key: 'device', label: 'Device', type: 'select', options: ['cpu', 'cuda:0', 'cuda:1'], default: 'cuda:0', desc: 'Training device' },
      { key: 'save_period', label: 'Save Period', type: 'number', min: 1, max: 50, default: 5, step: 1, desc: 'Save checkpoint every x epochs' },
      { key: 'workers', label: 'Workers', type: 'number', min: 0, max: 16, default: 4, step: 1, desc: 'Number of worker threads' },
      { key: 'pretrained', label: 'Use Pretrained', type: 'checkbox', default: true, desc: 'Use pretrained weights' },
      { key: 'seed', label: 'Random Seed', type: 'number', min: 0, max: 999999, default: 42, step: 1, desc: 'Random seed for reproducibility' },
      { key: 'dropout', label: 'Dropout', type: 'number', min: 0, max: 1, default: 0.0, step: 0.1, desc: 'Dropout rate' },
      { key: 'label_smoothing', label: 'Label Smoothing', type: 'number', min: 0, max: 1, default: 0.0, step: 0.1, desc: 'Label smoothing epsilon' }
    );

    // Advanced parameters group for YOLO
    baseGroups.push({
      group: PARAMETER_GROUPS.ADVANCED,
      params: [
        { key: 'split_ratio', label: 'Split Ratio', type: 'text', default: '[0.8, 0.2]', desc: 'Train/val split ratio (JSON array)' },
        { key: 'cache', label: 'Cache Images', type: 'checkbox', default: false, desc: 'Cache images for faster training' },
        { key: 'rect', label: 'Rectangular Training', type: 'checkbox', default: false, desc: 'Rectangular training' },
        { key: 'resume', label: 'Resume Path', type: 'text', default: '', desc: 'Resume training from checkpoint' },
        { key: 'amp', label: 'Mixed Precision', type: 'checkbox', default: true, desc: 'Use automatic mixed precision' },
        { key: 'single_cls', label: 'Single Class', type: 'checkbox', default: false, desc: 'Train as single-class dataset' },
        { key: 'cos_lr', label: 'Cosine LR', type: 'checkbox', default: false, desc: 'Use cosine LR scheduler' },
        { key: 'close_mosaic', label: 'Close Mosaic', type: 'number', min: 0, max: 20, default: 0, step: 1, desc: 'Disable mosaic augmentation' },
        { key: 'overlap_mask', label: 'Overlap Mask', type: 'checkbox', default: false, desc: 'Masks should overlap during training' },
      ]
    });
  } else if (algorithm === 'ResNet') {
    baseGroups[2].params.push(
      { key: 'resnet_depth', label: 'ResNet Depth', type: 'number', min: 18, max: 152, default: 50, step: 1 },
      { key: 'pretrained', label: 'Use Pretrained', type: 'checkbox', default: true }
    );
  }

  return baseGroups;
}; 