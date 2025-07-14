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
    fetchAllDatasets,
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
        
        // 3. Fetch all datasets
        console.log('\n3. Fetching all datasets...');
        const allDatasetsResult = await fetchAllDatasets();
        console.log('All datasets:', allDatasetsResult.data);
        
        // 4. Get dataset by ID
        console.log('\n4. Getting dataset by ID...');
        const datasetResult = await getDatasetById(1, 'raw');
        console.log('Dataset details:', datasetResult.data);
        
        // 5. Upload new dataset
        console.log('\n5. Uploading new dataset...');
        const uploadResult = await uploadDataset({
            name: 'Test Dataset',
            type: 'Image',
            size: '1.5GB'
        }, 'raw');
        console.log('Uploaded dataset:', uploadResult.data);
        
        // 6. Download dataset (will trigger browser download)
        console.log('\n6. Downloading dataset...');
        const downloadResult = await downloadDataset(1, 'raw');
        console.log('Download result:', downloadResult.message);
        
        // 7. Update dataset
        console.log('\n7. Updating dataset...');
        const updateResult = await updateDataset(1, { size: '3.0GB' }, 'raw');
        console.log('Updated dataset:', updateResult.data);
        
        // 8. Delete dataset (commented out to avoid deleting real data)
        // console.log('\n8. Deleting dataset...');
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