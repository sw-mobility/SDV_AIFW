const BASE_URL = 'http://localhost:5002';

/**
 * Start YOLO validation for detection task
 * POST /validation/yolo/validate/detection
 * 
 * @param {Object} params - Validation parameters
 * @param {string} params.uid - User ID (header)
 * @param {string} params.pid - Project ID
 * @param {string} params.tid - Training ID  
 * @param {string} params.cid - Codebase ID
 * @param {string} params.did - Dataset ID
 * @param {string} params.task_type - Task type (e.g., 'detection')
 * @param {Object} params.parameters - Model parameters
 * @returns {Promise<Object>} Validation start result with vid
 */
export async function startYoloValidation({ uid, pid, tid, cid, did, task_type, parameters }) {
    const response = await fetch(`${BASE_URL}/validation/yolo/validate/detection`, {
        method: 'POST',
        headers: { 
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({
            pid,
            tid,
            cid,
            did,
            task_type,
            parameters
        })
    });
    
    if (!response.ok) {
        let errorMessage = 'Failed to start validation';
        try {
            const errorData = await response.json();
            if (errorData.detail) {
                errorMessage = Array.isArray(errorData.detail) 
                    ? errorData.detail.map(d => d.msg).join(', ')
                    : errorData.detail;
            }
        } catch (e) {
            // If error response is not JSON, try to get text
            try {
                const errorText = await response.text();
                if (errorText) {
                    errorMessage = errorText;
                }
            } catch (textError) {
                // Use default error message
            }
        }
        throw new Error(errorMessage);
    }
    
    const result = await response.json();
    return result;
}

/**
 * Get validation status and results
 * GET /validation/yolo/{vid}
 * 
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
        let errorMessage = 'Failed to get validation status';
        try {
            const errorData = await response.json();
            if (errorData.detail) {
                errorMessage = Array.isArray(errorData.detail) 
                    ? errorData.detail.map(d => d.msg).join(', ')
                    : errorData.detail;
            }
        } catch (e) {
            // If error response is not JSON, try to get text
            try {
                const errorText = await response.text();
                if (errorText) {
                    errorMessage = errorText;
                }
            } catch (textError) {
                // Use default error message
            }
        }
        throw new Error(errorMessage);
    }
    
    const result = await response.json();
    return result;
}

/**
 * Get list of validation histories
 * GET /validation/list
 * 
 * @param {Object} params - Validation list parameters
 * @param {string} params.uid - User ID (header)
 * @returns {Promise<Array>} List of validation histories
 */
export async function getValidationList({ uid }) {
    const response = await fetch(`${BASE_URL}/validation/list`, {
        method: 'GET',
        headers: { 
            'accept': 'application/json',
            'uid': uid
        }
    });
    
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get validation list');
    }
    
    return await response.json();
}