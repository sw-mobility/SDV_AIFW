import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

// 상수들 (domain으로 분리할 필요 없는 단순한 상수들)
const METRIC_OPTIONS = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'f1', label: 'F1 Score' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall', label: 'Recall' },
];

const MOCK_MODELS = [
  { value: 'modelA', label: 'Model A' },
  { value: 'modelB', label: 'Model B' },
  { value: 'modelC', label: 'Model C' },
];

export const useValidation = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 데이터셋 목록 조회
  const fetchDatasets = useCallback(async () => {
    setDatasetLoading(true);
    setDatasetError(null);
    try {
      const res = await fetchLabeledDatasets({ uid });
      setDatasets(res.data || []);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  // 초기 로드
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // Validation 실행
  const handleRunValidation = useCallback(() => {
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
            model: MOCK_MODELS.find(m => m.value === selectedModel)?.label || selectedModel,
            dataset: selectedDataset?.name || '',
          }
        ]);
      }
    }, 300);
  }, [selectedModel, selectedMetric, selectedDataset]);

  return {
    // 상태
    selectedModel,
    setSelectedModel,
    selectedMetric,
    setSelectedMetric,
    selectedDataset,
    setSelectedDataset,
    status,
    progress,
    results,
    loading,
    error,
    datasets,
    datasetLoading,
    datasetError,
    
    // 핸들러
    handleRunValidation,
    
    // 상수
    metricOptions: METRIC_OPTIONS,
    mockModels: MOCK_MODELS
  };
}; 