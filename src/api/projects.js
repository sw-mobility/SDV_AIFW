const BASE_URL = 'http://localhost:5002';

export async function fetchProjects({ uid }) {
    const url = `${BASE_URL}/projects/projects/?uid=${encodeURIComponent(uid)}`;
    const response = await fetch(url);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch projects');
    }
    const data = await response.json();
    return { success: true, data, message: 'Projects fetched successfully' };
}

export async function createProject({ uid, name, description = '' }) {
    const requestBody = { uid, name, description };
    console.log('Sending request to:', `${BASE_URL}/projects/projects/create`);
    console.log('Request body:', requestBody);
    
    const response = await fetch(`${BASE_URL}/projects/projects/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
        const error = await response.text();
        console.error('Project creation failed:', response.status, error);
        throw new Error(error || `Failed to create project (${response.status})`);
    }

    const data = await response.json();
    console.log('Response data:', data);
    return { success: true, data, message: 'Project created successfully' };
}

export async function updateProject({ id, uid, name, description }) {
    const url = `${BASE_URL}/projects/projects/update?id=${encodeURIComponent(id)}`;
    const response = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid, name, description }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update project');
    }

    const data = await response.json();
    return { success: true, data, message: 'Project updated successfully' };
}

export async function deleteProject({ id, uid }) {
    const url = `${BASE_URL}/projects/projects/delete?id=${encodeURIComponent(id)}&uid=${encodeURIComponent(uid)}`;
    const response = await fetch(url, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to delete project');
    }

    return { success: true, message: 'Project deleted successfully' };
}

export async function getProjectById({ id, uid }) {
    const url = `${BASE_URL}/projects/projects/single/?id=${encodeURIComponent(id)}&uid=${encodeURIComponent(uid)}`;
    const response = await fetch(url);
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get project');
    }
    const data = await response.json();
    return { success: true, data, message: 'Project fetched successfully' };
} 