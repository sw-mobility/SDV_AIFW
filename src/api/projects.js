const BASE_URL = 'http://localhost:5002';

export async function fetchProjects({ uid }) {
    const response = await fetch(`${BASE_URL}/projects/projects/`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch projects');
    }
    const data = await response.json();
    return { success: true, data, message: 'Projects fetched successfully' };
}

export async function createProject({ uid, name, description = '' }) {
    const requestBody = { name, description };
    console.log('Sending request to:', `${BASE_URL}/projects/projects/create`);
    console.log('Request body:', requestBody);
    
    const response = await fetch(`${BASE_URL}/projects/projects/create`, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
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
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ name, description }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to update project');
    }

    const data = await response.json();
    return { success: true, data, message: 'Project updated successfully' };
}

export async function deleteProject({uid, id}) {
    const response = await fetch(`${BASE_URL}/projects/projects/?project_id=${encodeURIComponent(id)}`, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to delete project');
    }

    return { success: true, message: 'Project deleted successfully' };
}

export async function getProjectById({ id, uid }) {
    const response = await fetch(`${BASE_URL}/projects/projects/single/?id=${encodeURIComponent(id)}`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to get project');
    }
    const data = await response.json();
    return { success: true, data, message: 'Project fetched successfully' };
}

export async function getProjectByName({ name, uid }) {
    // 먼저 모든 프로젝트를 가져온 다음, name으로 필터링
    const response = await fetch(`${BASE_URL}/projects/projects/`, {
        headers: {
            'uid': uid
        }
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch projects');
    }
    const projects = await response.json();
    
    // name과 일치하는 프로젝트 찾기
    const project = projects.find(p => p.name === name);
    if (!project) {
        throw new Error(`Project with name '${name}' not found`);
    }
    
    return { success: true, data: project, message: 'Project fetched successfully' };
} 