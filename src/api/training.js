const BASE_URL = 'http://localhost:5002';

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

    const { uid, ...bodyData } = trainingData;
    
    console.log('Sending training request:', bodyData);
    
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
        console.error('Training API error:', error);
        throw new Error(error || 'YOLO training failed');
    }
    
    return await response.json();
}

/**
 * Get list of training histories
 * @param {string} uid - User ID
 * @returns {Promise<Array>} List of training histories
 */
export async function getTrainingList({ uid }) {
    const response = await fetch(`${BASE_URL}/training/list`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get training list');
    }
    
    return await response.json();
}

