const BASE_URL = 'http://localhost:5002';

// 파일명에서 경로를 제거하는 유틸리티 함수
function sanitizeFileName(file) {
    // 파일명에서 경로 부분을 제거하고 파일명만 추출
    const originalName = file.name || file.filename || 'unnamed';
    const fileName = originalName.split('/').pop().split('\\').pop();
    
    // 새로운 File 객체 생성 (파일명만 변경)
    return new File([file], fileName, {
        type: file.type,
        lastModified: file.lastModified
    });
}

export async function fetchRawDatasets({ uid }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch raw datasets');
    }
    const data = await response.json();
    return { success: true, data, message: 'Raw datasets fetched successfully' };
}

export async function fetchLabeledDatasets({ uid }) {
    const response = await fetch(`${BASE_URL}/datasets/labeled/`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch labeled datasets');
    }
    const data = await response.json();
    return { success: true, data, message: 'Labeled datasets fetched successfully' };
}



export async function deleteDatasets({ uid, target_did_list = [], target_path_list = [] }) {
    console.log('Delete datasets request:', { uid, target_did_list, target_path_list });
    
    const url = `${BASE_URL}/datasets/`;
    console.log('Delete URL:', url);
    
    const requestBody = { target_did_list, target_path_list };
    console.log('Request body:', requestBody);
    console.log('Request body JSON:', JSON.stringify(requestBody));
    
    const response = await fetch(url, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify(requestBody),
    });

    // 응답 텍스트 먼저 가져오기
    const responseText = await response.text();
    console.log('Delete response status:', response.status);
    console.log('Delete response text:', responseText);

    if (!response.ok) {
        console.error('Delete datasets error response:', {
            status: response.status,
            statusText: response.statusText,
            responseText
        });
        
        // Try to parse as JSON for better error handling
        let errorMessage = responseText;
        try {
            const errorJson = JSON.parse(responseText);
            if (errorJson.detail) {
                errorMessage = Array.isArray(errorJson.detail) 
                    ? errorJson.detail.map(d => d.msg).join(', ')
                    : errorJson.detail;
            } else if (errorJson.message) {
                errorMessage = errorJson.message;
            }
        } catch (e) {
            // If not JSON, use the text as is
            console.log('Response is not JSON, using as text');
        }
        
        // "삭제된 문서가 없습니다" 메시지 처리
        if (errorMessage.includes('삭제된 문서가 없습니다') || errorMessage.includes('No documents to delete')) {
            // 실제로는 삭제가 성공했을 가능성이 높음
            console.log('No documents to delete - this might indicate successful deletion');
            return { success: true, message: 'Datasets deleted successfully (no documents found to delete)' };
        }
        
        // 404 에러인 경우에도 삭제가 성공했을 가능성이 높음
        if (response.status === 404) {
            console.log('404 error - this might indicate successful deletion');
            return { success: true, message: 'Datasets deleted successfully (resources not found)' };
        }
        
        throw new Error(errorMessage || `Failed to delete datasets (${response.status})`);
    }

    // 성공 응답도 JSON으로 파싱 시도
    let result = { success: true, message: 'Datasets deleted successfully' };
    try {
        if (responseText.trim()) {
            const jsonResult = JSON.parse(responseText);
            result = { ...result, ...jsonResult };
            
            // 백엔드에서 성공 메시지가 포함된 경우
            if (jsonResult.message && (jsonResult.message.includes('삭제') || jsonResult.message.includes('delete'))) {
                result.message = jsonResult.message;
            }
        }
    } catch (e) {
        console.log('Success response is not JSON, using default success message');
    }

    return result;
}

export async function deleteData({ uid, target_did, target_name_list = [], target_path_list = [] }) {
    console.log('Delete data request params:', { uid, target_did, target_name_list, target_path_list });
    
    // API 문서에 맞춰 요청 본문 구성
    const requestBody = {
        target_did,
        target_name_list,
        target_path_list
    };
    
    console.log('Delete data request body:', requestBody);
    console.log('Delete data request body JSON:', JSON.stringify(requestBody, null, 2));
    
    const response = await fetch(`${BASE_URL}/datasets/data`, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify(requestBody),
    });

    // 응답 텍스트 먼저 가져오기
    const responseText = await response.text();
    console.log('Delete data response status:', response.status);
    console.log('Delete data response text:', responseText);

    if (!response.ok) {
        console.error('Delete data error response:', {
            status: response.status,
            statusText: response.statusText,
            responseText
        });
        
        // Try to parse as JSON for better error handling
        let errorMessage = responseText;
        try {
            const errorJson = JSON.parse(responseText);
            if (errorJson.detail) {
                errorMessage = Array.isArray(errorJson.detail) 
                    ? errorJson.detail.map(d => d.msg).join(', ')
                    : errorJson.detail;
            } else if (errorJson.message) {
                errorMessage = errorJson.message;
            }
        } catch (e) {
            // If not JSON, use the text as is
            console.log('Response is not JSON, using as text');
        }
        
        // "삭제된 문서가 없습니다" 메시지 처리
        if (errorMessage.includes('삭제된 문서가 없습니다') || errorMessage.includes('No documents to delete')) {
            // 실제로는 삭제가 성공했을 가능성이 높음
            console.log('No documents to delete - this might indicate successful deletion');
            return { success: true, message: 'Data deleted successfully (no documents found to delete)' };
        }
        
        // 404 에러인 경우에도 삭제가 성공했을 가능성이 높음
        if (response.status === 404) {
            console.log('404 error - this might indicate successful deletion');
            return { success: true, message: 'Data deleted successfully (resources not found)' };
        }
        
        throw new Error(errorMessage || `Failed to delete data (${response.status})`);
    }

    // 성공 응답도 JSON으로 파싱 시도
    let result = { success: true, message: 'Data deleted successfully' };
    try {
        if (responseText.trim()) {
            const jsonResult = JSON.parse(responseText);
            result = { ...result, ...jsonResult };
            
            // 백엔드에서 성공 메시지가 포함된 경우
            if (jsonResult.message && (jsonResult.message.includes('삭제') || jsonResult.message.includes('delete'))) {
                result.message = jsonResult.message;
            }
        }
    } catch (e) {
        console.log('Success response is not JSON, using default success message');
    }

    return result;
}

export async function uploadLabeledFiles({ files, uid, did }) {
    if (!did || !uid) {
        throw new Error('uploadLabeledFiles: did와 uid는 필수입니다. (did: ' + did + ', uid: ' + uid + ')');
    }
    const formData = new FormData();
    for (const file of files) {
        const sanitizedFile = sanitizeFileName(file);
        formData.append('files', sanitizedFile);
    }
    formData.append('did', did);
    const response = await fetch(`${BASE_URL}/datasets/labeled/upload`, {
        method: 'POST',
        headers: {
            'uid': uid
        },
        body: formData,
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to upload labeled files');
    }
    return await response.json();
}

export async function getLabeledDataset({ id, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/labeled/single?did=${encodeURIComponent(id)}`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get labeled dataset');
    }
    return await response.json();
}

export async function downloadDataset(datasetId, type = 'raw') {
    const response = await fetch(`${BASE_URL}/datasets/${type}/download?id=${encodeURIComponent(datasetId)}`);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to download dataset');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${datasetId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return { success: true, message: 'Dataset download started' };
}

export async function uploadDataset(datasetData, type = 'raw') {
    const response = await fetch(`${BASE_URL}/datasets/${type}/upload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(datasetData),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to upload dataset');
    }
    return await response.json();
}

export async function getDatasetById(datasetId, type = 'raw') {
    const response = await fetch(`${BASE_URL}/datasets/${type}/?id=${encodeURIComponent(datasetId)}`);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch dataset');
    }
    return await response.json();
}

export async function createRawDataset({ uid, name, description, type }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ name, description, type }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to create raw dataset');
    }
    return await response.json();
}

export async function updateRawDataset({ uid, did, name, description, type }) {
    if (!did || !uid) {
        throw new Error('updateRawDataset: did와 uid는 필수입니다. (did: ' + did + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/raw/`, {
        method: 'PUT',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ did, name, description, type}),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update raw dataset');
    }
    return await response.json();
}

export async function uploadRawFiles({ files, uid, did }) {
    if (!did || !uid) {
        throw new Error('uploadRawFiles: did와 uid는 필수입니다. (did: ' + did + ', uid: ' + uid + ')');
    }
    const formData = new FormData();
    for (const file of files) {
        const sanitizedFile = sanitizeFileName(file);
        formData.append('files', sanitizedFile);
    }
    formData.append('did', did);
    const response = await fetch(`${BASE_URL}/datasets/raw/upload`, {
        method: 'POST',
        headers: {
            'uid': uid
        },
        body: formData,
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to upload raw files');
    }
    return await response.json();
}

export async function getRawDataset({ id, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/single?did=${encodeURIComponent(id)}`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get raw dataset');
    }
    return await response.json();
}

export async function createLabeledDataset({ uid, name, description, type, task_type, label_format }) {
    const response = await fetch(`${BASE_URL}/datasets/labeled/`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ name, description, type, task_type, label_format }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to create labeled dataset');
    }
    return await response.json();
}

export async function updateLabeledDataset({ did, uid, name, description, type, task_type, label_format }) {
    if (!did || !uid) {
        throw new Error('updateLabeledDataset: did와 uid는 필수입니다. (did: ' + did + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/labeled/`, {
        method: 'PUT',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ did, name, description, type, task_type, label_format }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update labeled dataset');
    }
    return await response.json();
}

export async function downloadDatasetById({ uid, target_did, dataset_name }) {
    if (!target_did || !uid) {
        throw new Error('downloadDatasetById: target_did와 uid는 필수입니다. (target_did: ' + target_did + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/download-dataset`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ target_did })
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to download dataset');
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // dataset 이름을 파일명으로 사용, 없으면 target_did 사용
    const fileName = dataset_name ? `${dataset_name}.zip` : `${target_did}.zip`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    return true;
}

export async function downloadDataByPaths({ uid, target_path_list, dataset_name }) {
    if (!target_path_list || !uid) {
        throw new Error('downloadDataByPaths: target_path_list와 uid는 필수입니다. (target_path_list: ' + target_path_list + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/download-data`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ target_path_list })
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to download data');
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // dataset 이름을 파일명으로 사용, 없으면 기본값 사용
    const fileName = dataset_name ? `${dataset_name}_selected.zip` : `selected_data.zip`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    return true;
}

// 파일을 배치로 나누어 업로드하는 헬퍼 함수
const chunkArray = (array, chunkSize) => {
    const chunks = [];
    for (let i = 0; i < array.length; i += chunkSize) {
        chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
};

// 배치 업로드 함수 (raw files)
export async function uploadRawFilesInBatches({ files, uid, did, batchSize = 1000, onProgress }) {
    const batches = chunkArray(files, batchSize);
    const results = [];
    
    for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        const batchNumber = i + 1;
        const totalBatches = batches.length;
        
        // 진행률 콜백 호출
        if (onProgress) {
            onProgress({
                currentBatch: batchNumber,
                totalBatches,
                currentBatchSize: batch.length,
                totalFiles: files.length,
                uploadedFiles: i * batchSize
            });
        }
        
        try {
            const result = await uploadRawFiles({ files: batch, uid, did });
            results.push(result);
        } catch (error) {
            // 배치 업로드 실패 시 에러 정보와 함께 실패
            throw new Error(`Batch ${batchNumber} upload failed: ${error.message}`);
        }
    }
    
    return results;
}

// 배치 업로드 함수 (labeled files)
export async function uploadLabeledFilesInBatches({ files, uid, did, task_type, label_format, batchSize = 1000, onProgress }) {
    const batches = chunkArray(files, batchSize);
    const results = [];
    
    for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        const batchNumber = i + 1;
        const totalBatches = batches.length;
        
        // 진행률 콜백 호출
        if (onProgress) {
            onProgress({
                currentBatch: batchNumber,
                totalBatches,
                currentBatchSize: batch.length,
                totalFiles: files.length,
                uploadedFiles: i * batchSize
            });
        }
        
        try {
            const result = await uploadLabeledFiles({ files: batch, uid, did, task_type, label_format });
            results.push(result);
        } catch (error) {
            // 배치 업로드 실패 시 에러 정보와 함께 실패
            throw new Error(`Batch ${batchNumber} upload failed: ${error.message}`);
        }
    }
    
    return results;
}