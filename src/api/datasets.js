import {uid} from "./uid.js";

const BASE_URL = 'http://localhost:5002';

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
    const response = await fetch(`${BASE_URL}/datasets/`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, target_id_list, target_path_list }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to delete datasets');
    }

    return { success: true, message: 'Datasets deleted successfully' };
}

export async function deleteData({ uid, target_id_list = [], target_path_list = [] }) {
    const response = await fetch(`${BASE_URL}/datasets/data`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, target_id_list, target_path_list }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to delete data');
    }

    return { success: true, message: 'Data deleted successfully' };
}

export async function uploadLabeledFiles({ files, uid, id}) {
    const formData = new FormData();
    for (const file of files) formData.append('files', file);
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

export async function getLabeledDataset({ did, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/labeled/${encodeURIComponent(did)}?uid=${encodeURIComponent(uid)}`);
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, name, description, type }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to create raw dataset');
    }
    return await response.json();
}

export async function updateRawDataset({ did, name, description, type }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/?did=${encodeURIComponent(did)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, type }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update raw dataset');
    }
    return await response.json();
}

export async function uploadRawFiles({ files, uid, did }) {
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

export async function getRawDataset({ did, uid }) {
    const response = await fetch(`${BASE_URL}/datasets/raw/${encodeURIComponent(did)}?uid=${encodeURIComponent(uid)}`);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get raw dataset');
    }
    return await response.json();
}

export async function createLabeledDataset({ uid, name, description, type, task_type, label_format }) {
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

export async function updateLabeledDataset({ id, uid, name, description, type, task_type, label_format }) {
    if (!id || !uid) {
        throw new Error('updateLabeledDataset: id와 uid는 필수입니다. (id: ' + id + ', uid: ' + uid + ')');
    }
    const response = await fetch(`${BASE_URL}/datasets/labeled/?uid=${encodeURIComponent(uid)}&id=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, type, task_type, label_format }),
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update labeled dataset');
    }
    return await response.json();
}