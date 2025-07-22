// API Usage Examples
import { 
    fetchProjects, 
    createProject, 
    deleteProject, 
    updateProject, 
    getProjectById 
} from './projects.js';

import { 
    fetchRawDatasets, 
    fetchLabeledDatasets, 
    deleteDataset, 
    downloadDataset, 
    uploadDataset,
    getDatasetById,
    updateDataset
} from './datasets.js';

// Example: How to use the Projects API
export async function exampleProjectsAPI() {
    try {
        console.log('=== Projects API Examples ===');
        
        // 1. Fetch all projects
        console.log('1. Fetching all projects...');
        const projectsResult = await fetchProjects();
        console.log('Projects:', projectsResult.data);
        
        // 2. Create a new project
        console.log('\n2. Creating a new project...');
        const newProjectResult = await createProject({ name: 'Test Project' });
        console.log('New project:', newProjectResult.data);
        
        // 3. Get project by ID
        console.log('\n3. Getting project by ID...');
        const projectResult = await getProjectById(1);
        console.log('Project details:', projectResult.data);
        
        // 4. Update project
        console.log('\n4. Updating project...');
        const updateResult = await updateProject(1, { status: 'Training' });
        console.log('Updated project:', updateResult.data);
        
        // 5. Delete project (commented out to avoid deleting real data)
        // console.log('\n5. Deleting project...');
        // const deleteResult = await deleteProject(newProjectResult.data.id);
        // console.log('Delete result:', deleteResult.message);
        
    } catch (error) {
        console.error('Projects API Error:', error.message);
    }
}

// Example: How to use the Datasets API
export async function exampleDatasetsAPI() {
    try {
        console.log('\n=== Datasets API Examples ===');
        
        // 1. Fetch raw datasets
        console.log('1. Fetching raw datasets...');
        const rawDatasetsResult = await fetchRawDatasets();
        console.log('Raw datasets:', rawDatasetsResult.data);
        
        // 2. Fetch labeled datasets
        console.log('\n2. Fetching labeled datasets...');
        const labeledDatasetsResult = await fetchLabeledDatasets();
        console.log('Labeled datasets:', labeledDatasetsResult.data);
        
        // 3. Get dataset by ID
        console.log('\n3. Getting dataset by ID...');
        const datasetResult = await getDatasetById(1, 'raw');
        console.log('Dataset details:', datasetResult.data);
        
        // 4. Upload new dataset
        console.log('\n4. Uploading new dataset...');
        const uploadResult = await uploadDataset({
            name: 'Test Dataset',
            type: 'Image',
            size: '1.5GB'
        }, 'raw');
        console.log('Uploaded dataset:', uploadResult.data);
        
        // 5. Download dataset (will trigger browser download)
        console.log('\n5. Downloading dataset...');
        const downloadResult = await downloadDataset(1, 'raw');
        console.log('Download result:', downloadResult.message);
        
        // 6. Update dataset
        console.log('\n6. Updating dataset...');
        const updateResult = await updateDataset(1, { size: '3.0GB' }, 'raw');
        console.log('Updated dataset:', updateResult.data);
        
        // 7. Delete dataset (commented out to avoid deleting real data)
        // console.log('\n7. Deleting dataset...');
        // const deleteResult = await deleteDataset(uploadResult.data.id, 'raw');
        // console.log('Delete result:', deleteResult.message);
        
    } catch (error) {
        console.error('Datasets API Error:', error.message);
    }
}

// Example: Error handling
export async function exampleErrorHandling() {
    console.log('\n=== Error Handling Examples ===');
    
    try {
        // Try to get a non-existent project
        await getProjectById(999999);
    } catch (error) {
        console.log('Expected error for non-existent project:', error.message);
    }
    
    try {
        // Try to get a non-existent dataset
        await getDatasetById(999999, 'raw');
    } catch (error) {
        console.log('Expected error for non-existent dataset:', error.message);
    }
}

// Run all examples
export async function runAllExamples() {
    await exampleProjectsAPI();
    await exampleDatasetsAPI();
    await exampleErrorHandling();
}

// Uncomment to run examples in browser console
// runAllExamples().catch(console.error); 