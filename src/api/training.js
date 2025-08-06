const BASE_URL = 'http://localhost:5002';

export async function postYoloTraining({ uid, pid, task_type, parameters, dataset_id }) {
    const url = `${BASE_URL}/training/yolo`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, pid, task_type, parameters, dataset_id })
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'YOLO training failed');
    }
    return await response.json();
}

export async function postYoloTrainingResult({ uid, pid, status, task_type, parameters, dataset_id, artifact_path, error_details }) {
    const url = `${BASE_URL}/training/yolo/result`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, pid, status, task_type, parameters, dataset_id, artifact_path, error_details })
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'YOLO training result submission failed');
    }
    return await response.json();
}