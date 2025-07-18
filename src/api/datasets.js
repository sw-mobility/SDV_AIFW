// Mock API for Datasets
const MOCK_RAW_DATASETS = [
    {
        _id: 'raw1',
        uid: 'user1',
        did: 'did_raw1',
        name: 'Image Dataset 1',
        description: 'Sample image dataset',
        type: 'Image',
        total: 1200,
        path: '/datasets/raw1',
        created_at: '2024-01-15T10:00:00Z'
    },
    {
        _id: 'raw2',
        uid: 'user1',
        did: 'did_raw2',
        name: 'Text Dataset 1',
        description: 'Text data for NLP',
        type: 'Text',
        total: 800,
        path: '/datasets/raw2',
        created_at: '2024-01-14T09:00:00Z'
    },
    {
        _id: 'raw3',
        uid: 'user2',
        did: 'did_raw3',
        name: 'Audio Dataset 1',
        description: 'Audio samples',
        type: 'Audio',
        total: 500,
        path: '/datasets/raw3',
        created_at: '2024-01-13T08:00:00Z'
    }
];

const MOCK_LABELED_DATASETS = [
    {
        _id: 'labeled1',
        uid: 'user1',
        did: 'did_labeled1',
        name: 'Labeled Image Dataset 1',
        description: 'Labeled image data',
        type: 'Image',
        task_type: 'Classification',
        label_format: 'COCO',
        total: 1200,
        origin_raw: 'did_raw1',
        path: '/datasets/labeled1',
        created_at: '2024-01-16T11:00:00Z'
    },
    {
        _id: 'labeled2',
        uid: 'user1',
        did: 'did_labeled2',
        name: 'Labeled Text Dataset 1',
        description: 'Labeled text data',
        type: 'Text',
        task_type: 'NER',
        label_format: 'BIO',
        total: 800,
        origin_raw: 'did_raw2',
        path: '/datasets/labeled2',
        created_at: '2024-01-15T10:30:00Z'
    },
    {
        _id: 'labeled3',
        uid: 'user2',
        did: 'did_labeled3',
        name: 'Labeled Audio Dataset 1',
        description: 'Labeled audio data',
        type: 'Audio',
        task_type: 'Speech Recognition',
        label_format: 'TXT',
        total: 500,
        origin_raw: 'did_raw3',
        path: '/datasets/labeled3',
        created_at: '2024-01-14T09:30:00Z'
    }
];

// Simulate API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const BASE_URL = '';

export async function fetchRawDatasets({ uid, mockState } = {}) {
    if (BASE_URL) {
        // Real API mode
        const url = `${BASE_URL}/datasets/raw/?uid=${encodeURIComponent(uid)}`;
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to fetch raw datasets');
        }
        const data = await response.json();
        return { success: true, data, message: 'Raw datasets fetched successfully' };
    }
    // Mock mode (default)
    await delay(600);
    if (mockState?.error) throw new Error('Mock error!');
    if (mockState?.empty) return { success: true, data: [], message: 'No raw datasets' };
    return {
        success: true,
        data: MOCK_RAW_DATASETS,
        message: 'Raw datasets fetched successfully'
    };
}

export async function fetchLabeledDatasets({ uid, mockState } = {}) {
    if (BASE_URL) {
        const url = `${BASE_URL}/datasets/labeled/?uid=${encodeURIComponent(uid)}`;
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to fetch labeled datasets');
        }
        const data = await response.json();
        return { success: true, data, message: 'Labeled datasets fetched successfully' };
    }
    // Mock mode (default)
    await delay(600);
    if (mockState?.error) throw new Error('Mock error!');
    if (mockState?.empty) return { success: true, data: [], message: 'No labeled datasets' };
    return {
        success: true,
        data: MOCK_LABELED_DATASETS,
        message: 'Labeled datasets fetched successfully'
    };
}

export async function deleteLabeledDatasets({ uid, target_did_list }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/labeled/`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, target_did_list }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to delete labeled datasets');
        }
        return;
    }
    // Mock: just filter out from MOCK_LABELED_DATASETS
    for (const did of target_did_list) {
        const idx = MOCK_LABELED_DATASETS.findIndex(d => d.did === did || d.id === did);
        if (idx !== -1) MOCK_LABELED_DATASETS.splice(idx, 1);
    }
    await delay(300);
    return;
}

export async function uploadLabeledFiles({ files, uid, did, task_type, label_format }) {
    if (BASE_URL) {
        const formData = new FormData();
        for (const file of files) formData.append('files', file);
        formData.append('uid', uid);
        formData.append('did', did);
        formData.append('task_type', task_type);
        formData.append('label_format', label_format);
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
    // Mock: return fake file info
    await delay(1000);
    return [
        {
            _id: 'mockid', uid, did, dataset: did, name: files[0]?.name || 'mockfile', type: 'mock', task_type, file_format: 'csv', label_format, origin_raw: '', path: '/mock/path', created_at: new Date().toISOString()
        }
    ];
}

export async function getLabeledDataset({ did, uid }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/labeled/${encodeURIComponent(did)}?uid=${encodeURIComponent(uid)}`);
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to get labeled dataset');
        }
        return await response.json();
    }
    // Mock: find in MOCK_LABELED_DATASETS
    await delay(300);
    const dataset = MOCK_LABELED_DATASETS.find(d => d.did === did || d.id === did);
    if (!dataset) throw new Error('Dataset not found');
    return {
        ...dataset,
        data_list: [
            {
                _id: 'mockid', uid, did, dataset: did, name: 'mockfile', type: 'mock', task_type: 'mock', file_format: 'csv', label_format: 'mock', origin_raw: '', path: '/mock/path', created_at: new Date().toISOString()
            }
        ]
    };
}

export async function deleteLabeledData({ uid, did, target_id_list }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/labeled/data`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, did, target_id_list }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to delete labeled data');
        }
        return;
    }
    // Mock: just simulate
    await delay(300);
    return;
}

export async function fetchAllDatasets() {
    try {
        await delay(800);
        return {
            success: true,
            data: {
                raw: MOCK_RAW_DATASETS,
                labeled: MOCK_LABELED_DATASETS
            },
            message: 'All datasets fetched successfully'
        };
    } catch (error) {
        throw new Error('Failed to fetch datasets');
    }
}

export async function deleteDataset(datasetId, type = 'raw') {
    try {
        await delay(400);
        
        const datasetArray = type === 'raw' ? MOCK_RAW_DATASETS : MOCK_LABELED_DATASETS;
        const index = datasetArray.findIndex(dataset => dataset.id === datasetId);
        
        if (index === -1) {
            throw new Error('Dataset not found');
        }
        
        datasetArray.splice(index, 1);
        
        return {
            success: true,
            message: 'Dataset deleted successfully'
        };
    } catch (error) {
        throw new Error('Failed to delete dataset');
    }
}

export async function downloadDataset(datasetId, type = 'raw') {
    try {
        await delay(1000); // Simulate download time
        
        const datasetArray = type === 'raw' ? MOCK_RAW_DATASETS : MOCK_LABELED_DATASETS;
        const dataset = datasetArray.find(d => d.id === datasetId);
        
        if (!dataset) {
            throw new Error('Dataset not found');
        }
        
        // Simulate file download
        const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${dataset.name}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        return {
            success: true,
            message: 'Dataset download started'
        };
    } catch (error) {
        throw new Error('Failed to download dataset');
    }
}

export async function uploadDataset(datasetData, type = 'raw') {
    try {
        await delay(2000); // Simulate upload time
        const newDataset = {
            id: Date.now(),
            name: datasetData.name,
            type: datasetData.type,
            description: datasetData.description || '',
            size: datasetData.size,
            lastModified: new Date().toISOString().slice(0, 10),
            status: 'Active',
            ...(type === 'labeled' && { labelCount: datasetData.labelCount || 0 })
        };
        const datasetArray = type === 'raw' ? MOCK_RAW_DATASETS : MOCK_LABELED_DATASETS;
        datasetArray.unshift(newDataset);
        return {
            success: true,
            data: newDataset,
            message: 'Dataset uploaded successfully'
        };
    } catch (error) {
        throw new Error('Failed to upload dataset');
    }
}

export async function getDatasetById(datasetId, type = 'raw') {
    try {
        await delay(300);
        
        const datasetArray = type === 'raw' ? MOCK_RAW_DATASETS : MOCK_LABELED_DATASETS;
        const dataset = datasetArray.find(d => d.id === datasetId);
        
        if (!dataset) {
            throw new Error('Dataset not found');
        }
        
        return {
            success: true,
            data: dataset,
            message: 'Dataset fetched successfully'
        };
    } catch (error) {
        throw new Error('Failed to fetch dataset');
    }
}

export async function updateDataset(datasetId, updateData, type = 'raw') {
    try {
        await delay(500);
        
        const datasetArray = type === 'raw' ? MOCK_RAW_DATASETS : MOCK_LABELED_DATASETS;
        const dataset = datasetArray.find(d => d.id === datasetId);
        
        if (!dataset) {
            throw new Error('Dataset not found');
        }
        
        Object.assign(dataset, updateData, {
            lastModified: new Date().toISOString().slice(0, 10)
        });
        
        return {
            success: true,
            data: dataset,
            message: 'Dataset updated successfully'
        };
    } catch (error) {
        throw new Error('Failed to update dataset');
    }
} 

export async function createRawDataset({ uid, name, description, type }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/raw/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ uid, name, description, type }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to create raw dataset');
        }
        return await response.json();
    }
    // Mock: return a fake dataset in the same schema as the real API
    await delay(500);
    return {
        _id: 'mockid',
        uid,
        did: 'mockdid',
        name,
        description,
        type,
        total: 0,
        path: '/mock/path',
        created_at: new Date().toISOString()
    };
} 

export async function updateRawDataset({ did, name, description, type }) {
    const response = await fetch(`http://localhost/datasets/raw/?did=${encodeURIComponent(did)}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, description, type }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update raw dataset');
    }
    return await response.json();
} 

export async function deleteRawDatasets({ uid, target_did_list }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/raw/`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, target_did_list }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to delete raw datasets');
        }
        return;
    }
    // Mock: just filter out from MOCK_RAW_DATASETS
    for (const did of target_did_list) {
        const idx = MOCK_RAW_DATASETS.findIndex(d => d.did === did || d.id === did);
        if (idx !== -1) MOCK_RAW_DATASETS.splice(idx, 1);
    }
    await delay(300);
    return;
}

export async function uploadRawFiles({ files, uid, did }) {
    if (BASE_URL) {
        const formData = new FormData();
        for (const file of files) formData.append('files', file);
        formData.append('uid', uid);
        formData.append('did', did);
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
    // Mock: return fake file info
    await delay(1000);
    return [
        {
            _id: 'mockid', uid, did, dataset: did, name: files[0]?.name || 'mockfile', type: 'mock', file_format: 'csv', path: '/mock/path', created_at: new Date().toISOString()
        }
    ];
}

export async function getRawDataset({ did, uid }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/raw/${encodeURIComponent(did)}?uid=${encodeURIComponent(uid)}`);
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to get raw dataset');
        }
        return await response.json();
    }
    // Mock: find in MOCK_RAW_DATASETS
    await delay(300);
    const dataset = MOCK_RAW_DATASETS.find(d => d.did === did || d.id === did);
    if (!dataset) throw new Error('Dataset not found');
    return {
        ...dataset,
        data_list: [
            {
                _id: 'mockid', uid, did, dataset: did, name: 'mockfile', type: 'mock', file_format: 'csv', path: '/mock/path', created_at: new Date().toISOString()
            }
        ]
    };
} 

export async function deleteRawData({ uid, target_id_list }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/raw/data`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, target_id_list }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to delete raw data');
        }
        return;
    }
    // Mock: just simulate
    await delay(300);
    return;
}

export async function createLabeledDataset({ uid, name, description, type, task_type, label_format }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/labeled/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, name, description, type, task_type, label_format }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to create labeled dataset');
        }
        return await response.json();
    }
    // Mock: return fake labeled dataset
    await delay(500);
    return {
        _id: 'mockid', uid, did: 'mockdid', name, description, type, task_type, label_format, total: 0, origin_raw: '', path: '/mock/path', created_at: new Date().toISOString()
    };
}

export async function updateLabeledDataset({ did, uid, name, description, type, task_type, label_format }) {
    if (BASE_URL) {
        const response = await fetch(`${BASE_URL}/datasets/labeled/?did=${encodeURIComponent(did)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uid, name, description, type, task_type, label_format }),
        });
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Failed to update labeled dataset');
        }
        return await response.json();
    }
    // Mock: return updated labeled dataset
    await delay(500);
    return {
        _id: did, uid, did, name, description, type, task_type, label_format, total: 0, origin_raw: '', path: '/mock/path', created_at: new Date().toISOString()
    };
} 