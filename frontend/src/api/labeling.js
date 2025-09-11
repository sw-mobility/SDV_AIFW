import { API_BASE_URL } from './init.js';
import { uid } from './uid.js';

/**
 * YOLO detection labeling API
 * @param {Object} params - Labeling parameters
 * @param {string} params.pid - Project ID
 * @param {string} params.did - Dataset ID
 * @param {string} params.name - Labeling name
 * @param {string} params.cid - Configuration ID
 * @param {Object} params.parameters - YOLO parameters
 * @returns {Promise<string>} - Labeling result
 */
export const startYoloLabeling = async (params) => {
  try {
    const response = await fetch(`${API_BASE_URL}/labeling/yolo/detection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'uid': uid
      },
      body: JSON.stringify({
        pid: params.pid,
        did: params.did,
        name: params.name,
        cid: params.cid,
        parameters: params.parameters
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail?.[0]?.msg || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Failed to start YOLO labeling:', error);
    throw error;
  }
};

/**
 * Default YOLO parameters
 * API 스펙에 맞춰 모든 파라미터 포함
 */
export const DEFAULT_YOLO_PARAMS = {
  model: 'yolov8n.pt',
  source: '',
  conf: 0.25,
  iou: 0.45,
  imgsz: 640,
  rect: false,
  half: false,
  device: 'cpu',
  batch: 1,
  max_det: 300,
  vid_stride: 1,
  stream_buffer: false,
  agnostic_nms: false,
  classes: [],
  retina_masks: false,
  embed: [],
  project: 'runs/detect',
  name: 'exp',
  stream: false,
  save: true,
  save_txt: true,
  line_width: 3
};
