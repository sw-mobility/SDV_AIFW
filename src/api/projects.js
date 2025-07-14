// Mock API for Projects
let MOCK_PROJECTS = [
    { id: 1, name: 'AI Image Recognition', status: 'Active', lastModified: '2024-01-15' },
    { id: 2, name: 'Natural Language Processing', status: 'Active', lastModified: '2024-01-14' },
    { id: 3, name: 'Predictive Analytics', status: 'Active', lastModified: '2024-01-13' },
    { id: 4, name: 'Computer Vision Model', status: 'Training', lastModified: '2024-01-12' },
    { id: 5, name: 'Recommendation System', status: 'Active', lastModified: '2024-01-11' },
    { id: 6, name: 'Fraud Detection AI', status: 'Deployed', lastModified: '2024-01-10' },
    { id: 7, name: 'Sentiment Analysis', status: 'Active', lastModified: '2024-01-09' },
    { id: 8, name: 'Object Detection', status: 'Training', lastModified: '2024-01-08' },
    { id: 9, name: 'Text Classification', status: 'Active', lastModified: '2024-01-07' },
    { id: 10, name: 'Speech Recognition', status: 'Deployed', lastModified: '2024-01-06' },
    { id: 11, name: 'Data Mining Project', status: 'Active', lastModified: '2024-01-05' },
    { id: 12, name: 'Neural Network Model', status: 'Training', lastModified: '2024-01-04' },
];

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export async function fetchProjects() {
    await delay(300);
    return {
        success: true,
        data: [...MOCK_PROJECTS],
        message: 'Projects fetched successfully'
    };
}

export async function createProject(projectData) {
    await delay(300);
    const newProject = {
        id: Date.now(),
        name: projectData.name,
        status: 'Active',
        lastModified: new Date().toISOString().slice(0, 10),
    };
    MOCK_PROJECTS = [newProject, ...MOCK_PROJECTS];
    return {
        success: true,
        data: newProject,
        message: 'Project created successfully'
    };
}

export async function deleteProject(projectId) {
    await delay(200);
    MOCK_PROJECTS = MOCK_PROJECTS.filter(p => p.id !== projectId);
    return {
        success: true,
        message: 'Project deleted successfully'
    };
}

export async function updateProject(projectId, updateData) {
    await delay(200);
    let updated = null;
    MOCK_PROJECTS = MOCK_PROJECTS.map(p => {
        if (p.id === projectId) {
            updated = { ...p, ...updateData, lastModified: new Date().toISOString().slice(0, 10) };
            return updated;
        }
        return p;
    });
    return {
        success: true,
        data: updated,
        message: 'Project updated successfully'
    };
}

export async function getProjectById(projectId) {
    await delay(150);
    const project = MOCK_PROJECTS.find(p => p.id === projectId);
    if (!project) throw new Error('Project not found');
    return {
        success: true,
        data: project,
        message: 'Project fetched successfully'
    };
} 