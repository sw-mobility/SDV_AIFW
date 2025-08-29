/**
 * Labeling 파라미터 그룹 정의
 * YOLO detection API 스펙에 맞춰 구성
 */

export const LABELING_PARAM_GROUPS = [
  {
    title: 'Basic Settings',
    params: [
      {
        key: 'model',
        label: 'Model',
        type: 'text',
        defaultValue: 'yolo11n',
        description: 'Model file path or name',
        placeholder: 'yolo11n'
      },
      {
        key: 'device',
        label: 'Device',
        type: 'select',
        defaultValue: 'cpu',
        options: [
          { value: 'cpu', label: 'CPU' },
          { value: 'cuda', label: 'CUDA' },
          { value: '0', label: 'GPU 0' },
          { value: '1', label: 'GPU 1' }
        ],
        description: 'Device to run inference on'
      },
      {
        key: 'batch',
        label: 'Batch Size',
        type: 'number',
        defaultValue: 1,
        min: 1,
        max: 32,
        description: 'Batch size for inference'
      }
    ]
  },
  {
    title: 'Detection Settings',
    params: [
      {
        key: 'conf',
        label: 'Confidence Threshold',
        type: 'number',
        defaultValue: 0.25,
        min: 0,
        max: 1,
        step: 0.01,
        description: 'Confidence threshold for detections'
      },
      {
        key: 'iou',
        label: 'IoU Threshold',
        type: 'number',
        defaultValue: 0.45,
        min: 0,
        max: 1,
        step: 0.01,
        description: 'IoU threshold for NMS'
      },
      {
        key: 'imgsz',
        label: 'Image Size',
        type: 'number',
        defaultValue: 640,
        min: 32,
        max: 2048,
        description: 'Input image size'
      },
      {
        key: 'max_det',
        label: 'Max Detections',
        type: 'number',
        defaultValue: 300,
        min: 1,
        max: 1000,
        description: 'Maximum number of detections'
      },
      {
        key: 'line_width',
        label: 'Line Width',
        type: 'number',
        defaultValue: 3,
        min: 1,
        max: 10,
        description: 'Line width for bounding boxes'
      },
      {
        key: 'agnostic_nms',
        label: 'Agnostic NMS',
        type: 'boolean',
        defaultValue: false,
        description: 'Use agnostic NMS'
      },
      {
        key: 'rect',
        label: 'Rectangular',
        type: 'boolean',
        defaultValue: false,
        description: 'Use rectangular inference'
      },
      {
        key: 'half',
        label: 'Half Precision',
        type: 'boolean',
        defaultValue: false,
        description: 'Use FP16 half-precision inference'
      }
    ]
  },
  {
    title: 'Output Settings',
    params: [
      {
        key: 'project',
        label: 'Project Name',
        type: 'text',
        defaultValue: 'runs/labeling',
        description: 'Project name for output'
      },
      {
        key: 'name',
        label: 'Run Name',
        type: 'text',
        defaultValue: 'exp',
        description: 'Run name for output'
      },
      {
        key: 'save',
        label: 'Save Results',
        type: 'boolean',
        defaultValue: true,
        description: 'Save results'
      },
      {
        key: 'save_txt',
        label: 'Save Text',
        type: 'boolean',
        defaultValue: true,
        description: 'Save results to text file'
      },
      {
        key: 'save_conf',
        label: 'Save Confidence',
        type: 'boolean',
        defaultValue: false,
        description: 'Save confidence scores'
      },
      {
        key: 'save_crop',
        label: 'Save Crops',
        type: 'boolean',
        defaultValue: false,
        description: 'Save cropped detections'
      },
      {
        key: 'save_frames',
        label: 'Save Frames',
        type: 'boolean',
        defaultValue: false,
        description: 'Save video frames'
      }
    ]
  },
  {
    title: 'Visualization Settings',
    params: [
      {
        key: 'show_labels',
        label: 'Show Labels',
        type: 'boolean',
        defaultValue: true,
        description: 'Show labels on detections'
      },
      {
        key: 'show_conf',
        label: 'Show Confidence',
        type: 'boolean',
        defaultValue: true,
        description: 'Show confidence scores'
      },
      {
        key: 'show_boxes',
        label: 'Show Boxes',
        type: 'boolean',
        defaultValue: true,
        description: 'Show bounding boxes'
      },
      {
        key: 'show',
        label: 'Show Results',
        type: 'boolean',
        defaultValue: false,
        description: 'Show results'
      },
      {
        key: 'visualize',
        label: 'Visualize',
        type: 'boolean',
        defaultValue: false,
        description: 'Visualize features'
      },
      {
        key: 'augment',
        label: 'Augment',
        type: 'boolean',
        defaultValue: false,
        description: 'Apply augmentations'
      },
      {
        key: 'verbose',
        label: 'Verbose',
        type: 'boolean',
        defaultValue: false,
        description: 'Verbose output'
      }
    ]
  },
  {
    title: 'Advanced Settings',
    params: [
      {
        key: 'vid_stride',
        label: 'Video Stride',
        type: 'number',
        defaultValue: 1,
        min: 1,
        max: 100,
        description: 'Video frame stride'
      },
      {
        key: 'stream_buffer',
        label: 'Stream Buffer',
        type: 'boolean',
        defaultValue: false,
        description: 'Use stream buffer'
      },
      {
        key: 'stream',
        label: 'Stream',
        type: 'boolean',
        defaultValue: false,
        description: 'Stream results'
      },
      {
        key: 'retina_masks',
        label: 'Retina Masks',
        type: 'boolean',
        defaultValue: false,
        description: 'Use retina masks'
      },
      {
        key: 'classes',
        label: 'Classes',
        type: 'array',
        defaultValue: [0],
        description: 'Filter by class (array of integers)'
      },
      {
        key: 'embed',
        label: 'Embed',
        type: 'array',
        defaultValue: [0],
        description: 'Embed features (array of integers)'
      }
    ]
  }
];
