const BASE_URL = 'http://localhost:5002';

/**
 * Start YOLO validation for detection task
 * @param {string} uid - User ID (header)
 * @param {Object} params - Validation parameters
 * @param {string} params.pid - Project ID
 * @param {string} params.tid - Training ID
 * @param {string} params.cid - Codebase ID
 * @param {string} params.did - Dataset ID
 * @param {string} params.task_type - Task type (e.g., 'detection')
 * @param {Object} params.parameters - Model parameters
 * @param {string} params.parameters.model - Model file name
 * @param {number} params.parameters.imgsz - Image size
 * @param {number} params.parameters.batch - Batch size
 * @param {string} params.parameters.device - Device (cpu/gpu)
 * @param {number} params.parameters.workers - Number of workers
 * @param {number} params.parameters.conf - Confidence threshold
 * @param {number} params.parameters.iou - IoU threshold
 * @param {number} params.parameters.max_det - Maximum detections
 * @param {boolean} params.parameters.verbose - Verbose output
 * @param {boolean} params.parameters.half - Use half precision
 * @param {boolean} params.parameters.dnn - Use DNN
 * @param {boolean} params.parameters.agnostic_nms - Agnostic NMS
 * @param {boolean} params.parameters.augment - Test time augmentation
 * @param {boolean} params.parameters.rect - Rectangular inference
 * @returns {Promise<Object>} Validation start result
 */
export async function startYoloValidation({ uid, ...params }) {
    const response = await fetch(`${BASE_URL}/validation/yolo/validate/detection`, {
        method: 'POST',
        headers: { 
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify(params)
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to start validation');
    }
    
    return await response.json();
}

/**
 * Get validation status and results
 * @param {string} vid - Validation ID
 * @returns {Promise<Object>} Validation status and results
 */
export async function getValidationStatus({ vid }) {
    const response = await fetch(`${BASE_URL}/validation/yolo/${vid}`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json'
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get validation status');
    }
    
    return await response.json();
}
