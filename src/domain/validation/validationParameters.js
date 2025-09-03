/**
 * Validation 파라미터 그룹 정의
 * YOLO validation에 필요한 모든 파라미터들을 그룹별로 정리
 */

export const VALIDATION_PARAM_GROUPS = [
  {
    name: 'Model Settings',
    key: 'model',
    params: [
      {
        key: 'model',
        label: 'Model File',
        type: 'text',
        default: 'best.pt',
        desc: 'Path to the model file to use for validation',
        placeholder: 'best.pt',
        disabled: true
      },
      {
        key: 'task_type',
        label: 'Task Type',
        type: 'select',
        default: 'detection',
        options: ['detection', 'classification', 'segmentation'],
        desc: 'Type of task for validation',
        disabled: true
      },
      {
        key: 'device',
        label: 'Device',
        type: 'select',
        default: 'cpu',
        options: ['cpu', 'gpu'],
        desc: 'Device to run validation on (CPU or GPU)'
      }
    ]
  },
  {
    name: 'Input Settings',
    key: 'input',
    params: [
      {
        key: 'imgsz',
        label: 'Image Size',
        type: 'number',
        default: 640,
        min: 320,
        max: 1280,
        step: 32,
        desc: 'Input image size for validation'
      },
      {
        key: 'batch',
        label: 'Batch Size',
        type: 'number',
        default: 32,
        min: 1,
        max: 128,
        step: 1,
        desc: 'Batch size for validation'
      },
      {
        key: 'workers',
        label: 'Workers',
        type: 'number',
        default: 8,
        min: 0,
        max: 16,
        step: 1,
        desc: 'Number of worker threads for data loading'
      }
    ]
  },
  {
    name: 'Detection Settings',
    key: 'detection',
    params: [
      {
        key: 'conf',
        label: 'Confidence Threshold',
        type: 'number',
        default: 0.001,
        min: 0.0,
        max: 1.0,
        step: 0.001,
        desc: 'Confidence threshold for detections'
      },
      {
        key: 'iou',
        label: 'IoU Threshold',
        type: 'number',
        default: 0.6,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        desc: 'IoU threshold for NMS'
      },
      {
        key: 'max_det',
        label: 'Max Detections',
        type: 'number',
        default: 300,
        min: 1,
        max: 1000,
        step: 1,
        desc: 'Maximum number of detections per image'
      }
    ]
  },
  {
    name: 'Output Settings',
    key: 'output',
    params: [
      {
        key: 'save_json',
        label: 'Save JSON',
        type: 'checkbox',
        default: true,
        desc: 'Save results to JSON file'
      },
      {
        key: 'save_txt',
        label: 'Save TXT',
        type: 'checkbox',
        default: true,
        desc: 'Save results to TXT file'
      },
      {
        key: 'save_conf',
        label: 'Save Confidence',
        type: 'checkbox',
        default: true,
        desc: 'Save confidence scores'
      },
      {
        key: 'plots',
        label: 'Generate Plots',
        type: 'checkbox',
        default: true,
        desc: 'Generate validation plots'
      },
      {
        key: 'verbose',
        label: 'Verbose Output',
        type: 'checkbox',
        default: true,
        desc: 'Verbose output during validation'
      }
    ]
  },
  {
    name: 'Advanced Settings',
    key: 'advanced',
    params: [
      {
        key: 'half',
        label: 'Half Precision',
        type: 'checkbox',
        default: false,
        desc: 'Use half precision (FP16)'
      },
      {
        key: 'dnn',
        label: 'Use DNN',
        type: 'checkbox',
        default: false,
        desc: 'Use OpenCV DNN for ONNX inference'
      },
      {
        key: 'agnostic_nms',
        label: 'Agnostic NMS',
        type: 'checkbox',
        default: false,
        desc: 'Class-agnostic NMS'
      },
      {
        key: 'augment',
        label: 'Test Time Augmentation',
        type: 'checkbox',
        default: false,
        desc: 'Apply test time augmentation'
      },
      {
        key: 'rect',
        label: 'Rectangular Inference',
        type: 'checkbox',
        default: false,
        desc: 'Rectangular inference for better speed'
      }
    ]
  }
];

/**
 * 파라미터 값 정규화 함수
 * @param {any} value - 입력 값
 * @param {Object} param - 파라미터 정의
 * @returns {any} 정규화된 값
 */
export function normalizeValidationParamValue(value, param) {
  if (param.type === 'number') {
    const numValue = Number(value);
    if (isNaN(numValue)) return param.default;
    
    if (param.min !== undefined && numValue < param.min) return param.min;
    if (param.max !== undefined && numValue > param.max) return param.max;
    
    return numValue;
  }
  
  if (param.type === 'checkbox') {
    return Boolean(value);
  }
  
  return value;
}
