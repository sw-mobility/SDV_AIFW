const BASE_URL = 'http://localhost:5002';

/**
 * Get default YAML configuration for YOLO training
 * @param {string} uid - User ID
 * @param {string} pid - Project ID  
 * @param {string} dataset_id - Dataset ID
 * @returns {Promise<string>} Default YAML configuration
 */
export async function getYoloDefaultYaml({ uid, pid, dataset_id }) {
    const response = await fetch(`${BASE_URL}/training/yolo/default-yaml?pid=${encodeURIComponent(pid)}&dataset_id=${encodeURIComponent(dataset_id)}`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get default YAML configuration');
    }
    
    return await response.json();
}

/**
 * Get the custom YOLO model for training
 * @param {string} uid - User ID
 * @param {string} pid - Project ID
 * @param {string} tid - Training ID
 * @returns {Promise<string>} Custom YOLO model
 */
export async function getYoloCustomModel({ uid, pid, tid }) {
    const response = await fetch(`${BASE_URL}/training/yolo/custom-model?pid=${encodeURIComponent(pid)}&tid=${encodeURIComponent(tid)}`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get custom YOLO model');
    }
    
    return await response.json();
}

/**
 * Get the custom codebase for YOLO training
 * @param {string} uid - User ID
 * @param {string} pid - Project ID
 * @param {string} codebase_id - Codebase ID
 * @returns {Promise<string>} Custom codebase
 */
export async function getYoloCustomCodebase({ uid, pid, codebase_id }) {
    const response = await fetch(`${BASE_URL}/training/yolo/custom-codebase?pid=${encodeURIComponent(pid)}&codebase_id=${encodeURIComponent(codebase_id)}`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get custom codebase');
    }
    
    return await response.json();
}

/**
 * Create a snapshot of the current training state
 * @param {string} uid - User ID
 * @param {string} pid - Project ID
 * @param {string} name - Snapshot name
 * @param {string} algorithm - Algorithm (e.g., 'yolo')
 * @param {string} task_type - Task type (e.g., 'detection')
 * @param {string|null} description - Optional description
 * @returns {Promise<string>} Snapshot creation result
 */
export async function createYoloSnapshot({ uid, pid, name, algorithm, task_type, description = null }) {
    const response = await fetch(`${BASE_URL}/training/yolo/snapshot?pid=${encodeURIComponent(pid)}&name=${encodeURIComponent(name)}&algorithm=${encodeURIComponent(algorithm)}&task_type=${encodeURIComponent(task_type)}${description ? `&description=${encodeURIComponent(description)}` : ''}`, {
        method: 'POST',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to create snapshot');
    }
    
    return await response.json();
}

/**
 * Start YOLO training with the provided parameters
 * @param {Object} trainingData - Training configuration
 * @param {string} trainingData.uid - User ID
 * @param {string} trainingData.pid - Project ID
 * @param {string} trainingData.task_type - Task type (e.g., 'detection')
 * @param {string} trainingData.codebase_id - Codebase ID
 * @param {Object} trainingData.parameters - Training parameters
 * @param {string} trainingData.parameters.model - Model name (e.g., 'yolo11n')
 * @param {number[]} trainingData.parameters.split_ratio - Dataset split ratio [train, val, test]
 * @param {number} trainingData.parameters.epochs - Number of training epochs
 * @param {string} trainingData.dataset_id - Dataset ID
 * @returns {Promise<string>} Training result
 */
export async function postYoloTraining(trainingData) {
    const url = `${BASE_URL}/training/yolo/detection`;
    
    // uid를 body에서 제거하고 header로 이동
    const { uid, ...bodyData } = trainingData;
    
    const response = await fetch(url, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'accept': 'application/json',
            'uid': uid
        },
        body: JSON.stringify(bodyData)
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'YOLO training failed');
    }
    
    return await response.json();
}

