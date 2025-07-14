import React, { useState } from 'react';
import { runAllExamples } from '../api/api-examples.js';
import { fetchProjects, createProject } from '../api/projects.js';
import { fetchRawDatasets, uploadDataset } from '../api/datasets.js';
import Card from '../components/ui/Card.jsx';
import Button from '../components/ui/Button.jsx';

const ApiTestPage = () => {
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const addResult = (title, data) => {
        setResults(prev => [...prev, { title, data, timestamp: new Date().toLocaleTimeString() }]);
    };

    const clearResults = () => {
        setResults([]);
    };

    const testProjectsAPI = async () => {
        setLoading(true);
        try {
            addResult('Fetching Projects', 'Loading...');
            const projectsResult = await fetchProjects();
            addResult('Projects Result', projectsResult.data);

            addResult('Creating Project', 'Loading...');
            const createResult = await createProject({ name: 'Test Project ' + Date.now() });
            addResult('Create Project Result', createResult.data);
        } catch (error) {
            addResult('Error', error.message);
        } finally {
            setLoading(false);
        }
    };

    const testDatasetsAPI = async () => {
        setLoading(true);
        try {
            addResult('Fetching Raw Datasets', 'Loading...');
            const datasetsResult = await fetchRawDatasets();
            addResult('Raw Datasets Result', datasetsResult.data);

            addResult('Uploading Dataset', 'Loading...');
            const uploadResult = await uploadDataset({
                name: 'Test Dataset ' + Date.now(),
                type: 'Image',
                size: '1.5GB'
            }, 'raw');
            addResult('Upload Dataset Result', uploadResult.data);
        } catch (error) {
            addResult('Error', error.message);
        } finally {
            setLoading(false);
        }
    };

    const runAllTests = async () => {
        setLoading(true);
        try {
            // Clear console and run examples
            console.clear();
            await runAllExamples();
            addResult('All Tests', 'Check browser console for detailed results');
        } catch (error) {
            addResult('Error', error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
            <h1>API Test Page</h1>
            <p>이 페이지에서 Mock API들을 테스트할 수 있습니다.</p>

            <div style={{ marginBottom: '2rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <Button 
                    onClick={testProjectsAPI} 
                    disabled={loading}
                    style={{ minWidth: '150px' }}
                >
                    Test Projects API
                </Button>
                <Button 
                    onClick={testDatasetsAPI} 
                    disabled={loading}
                    style={{ minWidth: '150px' }}
                >
                    Test Datasets API
                </Button>
                <Button 
                    onClick={runAllTests} 
                    disabled={loading}
                    style={{ minWidth: '150px' }}
                >
                    Run All Tests
                </Button>
                <Button 
                    onClick={clearResults}
                    variant="secondary"
                    style={{ minWidth: '150px' }}
                >
                    Clear Results
                </Button>
            </div>

            {loading && (
                <div style={{ 
                    padding: '1rem', 
                    backgroundColor: '#f0f8ff', 
                    borderRadius: '8px',
                    marginBottom: '1rem'
                }}>
                    Loading...
                </div>
            )}

            <div style={{ display: 'grid', gap: '1rem' }}>
                {results.map((result, index) => (
                    <Card key={index} style={{ padding: '1rem' }}>
                        <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            marginBottom: '0.5rem'
                        }}>
                            <h3 style={{ margin: 0 }}>{result.title}</h3>
                            <small style={{ color: '#666' }}>{result.timestamp}</small>
                        </div>
                        <pre style={{ 
                            backgroundColor: '#f5f5f5', 
                            padding: '1rem', 
                            borderRadius: '4px',
                            overflow: 'auto',
                            maxHeight: '300px'
                        }}>
                            {typeof result.data === 'object' 
                                ? JSON.stringify(result.data, null, 2)
                                : result.data
                            }
                        </pre>
                    </Card>
                ))}
            </div>

            {results.length === 0 && !loading && (
                <Card style={{ 
                    padding: '2rem', 
                    textAlign: 'center',
                    color: '#666'
                }}>
                    <p>테스트를 실행하면 결과가 여기에 표시됩니다.</p>
                    <p>자세한 로그는 브라우저 콘솔을 확인하세요.</p>
                </Card>
            )}
        </div>
    );
};

export default ApiTestPage; 