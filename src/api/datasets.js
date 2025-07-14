// Mock API for Datasets
const MOCK_RAW_DATASETS = [
    { id: 1, name: 'Image Dataset 1', type: 'Image', size: '2.3GB', lastModified: '2024-01-15', status: 'Active' },
    { id: 2, name: 'Image Dataset 2', type: 'Image', size: '1.8GB', lastModified: '2024-01-14', status: 'Active' },
    { id: 3, name: 'Text Dataset 1', type: 'Text', size: '500MB', lastModified: '2024-01-13', status: 'Active' },
    { id: 4, name: 'Audio Dataset 1', type: 'Audio', size: '3.2GB', lastModified: '2024-01-12', status: 'Active' },
    { id: 5, name: 'Video Dataset 1', type: 'Video', size: '5.1GB', lastModified: '2024-01-11', status: 'Active' },
    { id: 6, name: 'Tabular Dataset 1', type: 'Tabular', size: '150MB', lastModified: '2024-01-10', status: 'Active' },
    { id: 7, name: 'Time Series Dataset 1', type: 'TimeSeries', size: '800MB', lastModified: '2024-01-09', status: 'Active' },
    { id: 8, name: 'Graph Dataset 1', type: 'Graph', size: '2.1GB', lastModified: '2024-01-08', status: 'Active' },
];

const MOCK_LABELED_DATASETS = [
    { id: 1, name: 'Labeled Image Dataset 1', type: 'Image', size: '2.8GB', lastModified: '2024-01-15', status: 'Active', labelCount: 15000 },
    { id: 2, name: 'Labeled Text Dataset 1', type: 'Text', size: '1.5GB', lastModified: '2024-01-14', status: 'Active', labelCount: 8000 },
    { id: 3, name: 'Labeled Audio Dataset 1', type: 'Audio', size: '3.5GB', lastModified: '2024-01-13', status: 'Active', labelCount: 12000 },
    { id: 4, name: 'Labeled Video Dataset 1', type: 'Video', size: '6.2GB', lastModified: '2024-01-12', status: 'Active', labelCount: 5000 },
    { id: 5, name: 'Labeled Image Dataset 2', type: 'Image', size: '1.9GB', lastModified: '2024-01-11', status: 'Active', labelCount: 9500 },
    { id: 6, name: 'Labeled Tabular Dataset 1', type: 'Tabular', size: '200MB', lastModified: '2024-01-10', status: 'Active', labelCount: 25000 },
    { id: 7, name: 'Labeled Time Series Dataset 1', type: 'TimeSeries', size: '1.2GB', lastModified: '2024-01-09', status: 'Active', labelCount: 18000 },
    { id: 8, name: 'Labeled Graph Dataset 1', type: 'Graph', size: '2.5GB', lastModified: '2024-01-08', status: 'Active', labelCount: 7500 },
];

// Simulate API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export async function fetchRawDatasets(mockState) {
    await delay(600);
    if (mockState?.error) throw new Error('Mock error!');
    if (mockState?.empty) return { success: true, data: [], message: 'No raw datasets' };
    return {
        success: true,
        data: MOCK_RAW_DATASETS,
        message: 'Raw datasets fetched successfully'
    };
}

export async function fetchLabeledDatasets(mockState) {
    await delay(600);
    if (mockState?.error) throw new Error('Mock error!');
    if (mockState?.empty) return { success: true, data: [], message: 'No labeled datasets' };
    return {
        success: true,
        data: MOCK_LABELED_DATASETS,
        message: 'Labeled datasets fetched successfully'
    };
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