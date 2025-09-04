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
    const url = `${BASE_URL}/projects/projects/update?pid=${encodeURIComponent(id)}`;
    const response = await fetch(url, {
        method: 'PUT',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
        body: JSON.stringify({ pid: id, name, description }),
    });

    if (!response.ok) {
        let errorMessage = 'Failed to update project';
        
        try {
            if (response.status === 422) {
                // Validation Error 처리
                const errorData = await response.json();
                console.log('Update validation error data:', errorData);
                errorMessage = errorData.detail?.[0]?.msg || 'Validation error occurred';
            } else {
                const errorText = await response.text();
                console.log('Update error response text:', errorText);
                
                try {
                    const errorJson = JSON.parse(errorText);
                    if (errorJson.detail) {
                        errorMessage = errorJson.detail;
                    }
                } catch {
                    errorMessage = errorText || `HTTP ${response.status}: ${response.statusText}`;
                }
            }
        } catch (parseError) {
            console.error('Error parsing update response:', parseError);
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
    }

    const data = await response.json();
    console.log('Update project success response:', data);
    return { success: true, data, message: 'Project updated successfully' };
}

export async function deleteProject({uid, id}) {
    const url = `${BASE_URL}/projects/projects/?pid=${encodeURIComponent(id)}`;
    console.log('Delete project request:', { url, uid, id });
    
    const response = await fetch(url, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'uid': uid
        },
    });

    console.log('Delete project response:', { 
        status: response.status, 
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
    });

    if (!response.ok) {
        let errorMessage = 'Failed to delete project';
        
        try {
            if (response.status === 422) {
                // Validation Error 처리
                const errorData = await response.json();
                console.log('Validation error data:', errorData);
                errorMessage = errorData.detail?.[0]?.msg || 'Validation error occurred';
            } else {
                // 다른 에러들 처리
                const errorText = await response.text();
                console.log('Error response text:', errorText);
                
                try {
                    const errorJson = JSON.parse(errorText);
                    if (errorJson.detail) {
                        errorMessage = errorJson.detail;
                    }
                } catch {
                    errorMessage = errorText || `HTTP ${response.status}: ${response.statusText}`;
                }
            }
        } catch (parseError) {
            console.error('Error parsing response:', parseError);
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
    }

    // 성공 시 응답 데이터 처리 (API 명세에 따르면 string 반환)
    const responseData = await response.text();
    console.log('Delete project success response:', responseData);
    return { success: true, data: responseData, message: 'Project deleted successfully' };
}

export async function getProjectById({ id, uid }) {
    const response = await fetch(`${BASE_URL}/projects/projects/single/?pid=${encodeURIComponent(id)}`, {
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