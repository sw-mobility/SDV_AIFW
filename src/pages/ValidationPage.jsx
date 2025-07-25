import React, { useState, useEffect } from 'react';
import styles from './ValidationPage.module.css';
import dsStyles from '../components/training/DatasetSelector.module.css';
import ModelSelector from '../components/optimization/ModelSelector.jsx';
import DatasetSelector from '../components/training/DatasetSelector.jsx';
import ProgressBar from '../components/common/ProgressBar.jsx';
import Button from '../components/common/Button.jsx';
import { fetchLabeledDatasets } from '../api/datasets.js';
import { uid } from '../api/uid.js';

const metricOptions = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'f1', label: 'F1 Score' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall', label: 'Recall' },
];

const mockModels = [
  { value: 'modelA', label: 'Model A' },
  { value: 'modelB', label: 'Model B' },
  { value: 'modelC', label: 'Model C' },
];

const ValidationPage = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  useEffect(() => {
    setDatasetLoading(true);
    fetchLabeledDatasets({ uid })
      .then(res => {
        setDatasets(res.data || []);
        setDatasetError(null);
      })
      .catch(e => setDatasetError(e.message))
      .finally(() => setDatasetLoading(false));
  }, []);

  // Simulate API call for validation
  const handleRunValidation = () => {
    setStatus('running');
    setProgress(0);
    setError(null);
    setLoading(true);
    let pct = 0;
    const interval = setInterval(() => {
      pct += 10;
      setProgress(pct);
      if (pct >= 100) {
        clearInterval(interval);
        setStatus('success');
        setLoading(false);
        const value = (Math.random() * 0.5 + 0.5).toFixed(3);
        setResults(prev => [
          ...prev,
          {
            metric: selectedMetric,
            value,
            model: mockModels.find(m => m.value === selectedModel)?.label || selectedModel,
            dataset: selectedDataset?.name || '',
          }
        ]);
      }
    }, 300);
  };

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Validation</h1>
        <p className={styles.pageDescription}>
          Select a model, metric, and dataset to run validation. Results will be saved and displayed below.
        </p>
      </div>
      {error && (
        <div className={styles.errorMessage}>
          <span>Error: {error}</span>
        </div>
      )}
      <div className={styles.sectionWrap}>
        {/* 좌측: 모델/데이터셋/메트릭 선택 패널 */}
        <div className={styles.leftPanel}>
          <div className={styles.selectorBlock}>
            <ModelSelector value={selectedModel} onChange={setSelectedModel} disabled={status === 'running'} showInfo={false} />
          </div>
          <div className={styles.selectorBlock}>
            <div className={dsStyles.selectorBox}>
              <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Metric</label>
              <select
                className={dsStyles.select}
                value={selectedMetric}
                onChange={e => setSelectedMetric(e.target.value)}
                disabled={status === 'running'}
              >
                <option value="">Select metric</option>
                {metricOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className={styles.selectorBlock}>
            <DatasetSelector
              datasets={datasets}
              selectedDataset={selectedDataset}
              onDatasetChange={setSelectedDataset}
              datasetLoading={datasetLoading}
              datasetError={datasetError}
            />
          </div>
        </div>
        {/* 우측: 실행 및 결과 패널 */}
        <div className={styles.rightPanel}>
          <div className={styles.workspace}>
            {/* 헤더 섹션 */}
            <div className={styles.header}>
              <div>
                <h2 className={styles.pageTitle} style={{ fontSize: 22, marginBottom: 0 }}>Validation Execution</h2>
              </div>
              <div className={styles.statusBadge}>
                {status === 'idle' && <span className={styles.statusIdle}>Ready</span>}
                {status === 'running' && <span className={styles.statusRunning}>Running</span>}
                {status === 'success' && <span className={styles.statusSuccess}>Completed</span>}
                {status === 'error' && <span className={styles.statusError}>Failed</span>}
              </div>
            </div>
            {/* 실행 버튼 */}
            <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="primary"
                size="large"
                onClick={handleRunValidation}
                disabled={
                  !selectedModel || !selectedMetric || !selectedDataset || status === 'running'
                }
                className={styles.runButton}
              >
                {status === 'running' ? 'Running...' : 'Run Validation'}
              </Button>
            </div>
            {/* 진행률 표시 */}
            {status !== 'idle' && (
              <div className={styles.progressSection}>
                <ProgressBar
                  percentage={progress}
                  status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
                  completeText="Validation completed!"
                />
              </div>
            )}
            {/* 결과 표 */}
            {results.length > 0 && (
              <div style={{ marginTop: 32 }}>
                <h3>Results</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
                      <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Dataset</th>
                      <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Metric</th>
                      <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '10px' }}>{r.model}</td>
                        <td style={{ padding: '10px' }}>{r.dataset}</td>
                        <td style={{ padding: '10px' }}>{r.metric}</td>
                        <td style={{ padding: '10px' }}>{r.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValidationPage;



