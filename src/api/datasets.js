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
    const url = `${BASE_URL}/datasets/raw/?uid=${encodeURIComponent(uid)}`;
    const response = await fetch(url);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch raw datasets');
    }
    const data = await response.json();
    return { success: true, data, message: 'Raw datasets fetched successfully' };
}

export async function fetchLabeledDatasets({ uid }) {
    const url = `${BASE_URL}/datasets/labeled/?uid=${encodeURIComponent(uid)}`;
    const response = await fetch(url);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch labeled datasets');
    }
    const data = await response.json();
    return { success: true, data, message: 'Labeled datasets fetched successfully' };
}

export async function deleteDatasets({ uid, target_id_list = [], target_path_list = [] }) {
    console.log('Delete datasets request:', { uid, target_id_list, target_path_list });
    
    const url = `${BASE_URL}/datasets/`;
    console.log('Delete URL:', url);
    
    const requestBody = { target_id_list, target_path_list };
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

    if (!response.ok) {
        const errorText = await response.text();
        console.error('Delete datasets error response:', {
            status: response.status,
            statusText: response.statusText,
            errorText
        });
        
        // Try to parse as JSON for better error handling
        let errorMessage = errorText;
        try {
            const errorJson = JSON.parse(errorText);
            if (errorJson.detail) {
                errorMessage = Array.isArray(errorJson.detail) 
                    ? errorJson.detail.map(d => d.msg).join(', ')
                    : errorJson.detail;
            }
        } catch (e) {
            // If not JSON, use the text as is
        }
        
        throw new Error(errorMessage || `Failed to delete datasets (${response.status})`);
    }

    return { success: true, message: 'Datasets deleted successfully' };
}

export async function deleteData({ uid, target_id_list = [], target_path_list = [] }) {
    const response = await fetch(`${BASE_URL}/datasets/data`, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ target_id_list, target_path_list }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to delete data');
    }

    return { success: true, message: 'Data deleted successfully' };
}

export async function uploadLabeledFiles({ files, uid, id }) {
    const formData = new FormData();
    for (const file of files) {
        const sanitizedFile = sanitizeFileName(file);
        formData.append('files', sanitizedFile);
    }
    formData.append('uid', uid);
    formData.append('id', id);
    const response = await fetch(`${BASE_URL}/datasets/labeled/upload`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to upload labeled files');
    }
    return await response.json();
}

export async function getLabeledDataset({ id, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/labeled/single/?id=${encodeURIComponent(id)}&uid=${encodeURIComponent(uid)}`);
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

export async function updateRawDataset({ uid, id, name, description, type }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/?id=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ name, description, type}),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update raw dataset');
    }
    return await response.json();
}

export async function uploadRawFiles({ files, uid, id }) {
    const formData = new FormData();
    for (const file of files) {
        const sanitizedFile = sanitizeFileName(file);
        formData.append('files', sanitizedFile);
    }
    formData.append('uid', uid);
    formData.append('id', id);
    const response = await fetch(`${BASE_URL}/datasets/raw/upload`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to upload raw files');
    }
    return await response.json();
}

export async function getRawDataset({ id, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/single/?id=${encodeURIComponent(id)}&uid=${encodeURIComponent(uid)}`);
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

export async function updateLabeledDataset({ id, uid, name, description, type, task_type, label_format }) {
    if (!id || !uid) {
        throw new Error('updateLabeledDataset: id와 uid는 필수입니다. (id: ' + id + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/labeled/?id=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ name, description, type, task_type, label_format }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update labeled dataset');
    }
    return await response.json();
}

export async function downloadDatasetById({ uid, target_id, dataset_name }) {
    const response = await fetch(`${BASE_URL}/datasets/download-dataset`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ target_id })
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to download dataset');
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // dataset_name이 있으면 사용하고, 없으면 target_id 사용
    const fileName = dataset_name ? `${dataset_name}.zip` : `${target_id}.zip`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    return true;
}

export async function downloadDataByPaths({ uid, target_path_list, dataset_name }) {
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
    // dataset_name이 있으면 사용하고, 없으면 기본 이름 사용
    const fileName = dataset_name ? `${dataset_name}_selected_data.zip` : `data.zip`;
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
export async function uploadRawFilesInBatches({ files, uid, id, batchSize = 1000, onProgress }) {
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
            const result = await uploadRawFiles({ files: batch, uid, id });
            results.push(result);
        } catch (error) {
            // 배치 업로드 실패 시 에러 정보와 함께 실패
            throw new Error(`Batch ${batchNumber} upload failed: ${error.message}`);
        }
    }
    
    return results;
}

// 배치 업로드 함수 (labeled files)
export async function uploadLabeledFilesInBatches({ files, uid, id, task_type, label_format, batchSize = 1000, onProgress }) {
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
            const result = await uploadLabeledFiles({ files: batch, uid, id, task_type, label_format });
            results.push(result);
        } catch (error) {
            // 배치 업로드 실패 시 에러 정보와 함께 실패
            throw new Error(`Batch ${batchNumber} upload failed: ${error.message}`);
        }
    }
    
    return results;
}