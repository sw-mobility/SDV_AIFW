import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { startYoloValidation, getValidationStatus } from '../../api/validation.js';
import { uid } from '../../api/uid.js';

// 상수들 (domain으로 분리할 필요 없는 단순한 상수들)
const MOCK_MODELS = [
  { value: 'best.pt', label: 'Best Model (best.pt)' },
  { value: 'last.pt', label: 'Last Model (last.pt)' },
  { value: 'custom.pt', label: 'Custom Model' },
];

const DEVICE_OPTIONS = [
  { value: 'cpu', label: 'CPU' },
  { value: 'gpu', label: 'GPU' },
];

export const useValidation = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // YOLO validation parameters (간소화된 파라미터 세트)
  const [validationParams, setValidationParams] = useState({
    model: 'best.pt',
    imgsz: 640,
    batch: 32,
    device: 'cpu',
    workers: 8,
    conf: 0.001,
    iou: 0.6,
    max_det: 300,
    verbose: true,
    half: false,
    dnn: false,
    agnostic_nms: false,
    augment: false,
    rect: false
  });
  
  // Validation ID for polling
  const [currentVid, setCurrentVid] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(null);

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

  // 모델 선택 시 validation 파라미터 업데이트
  const handleModelChange = useCallback((model) => {
    setSelectedModel(model);
    if (model) {
      setValidationParams(prev => ({ ...prev, model }));
    }
  }, []);

  // Validation 상태 폴링
  const pollValidationStatus = useCallback(async (vid) => {
    try {
      const result = await getValidationStatus({ vid });
      
      if (result.status === 'completed') {
        setStatus('success');
        setProgress(100);
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
        
        // 결과 추가
        setResults(prev => [
          ...prev,
          {
            vid: result.vid,
            model: selectedModel,
            dataset: selectedDataset?.name || '',
            timestamp: new Date().toISOString(),
            status: result.status,
            results: result.results || {}
          }
        ]);
      } else if (result.status === 'failed') {
        setStatus('error');
        setError(result.error || 'Validation failed');
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
      } else if (result.status === 'running') {
        // 진행률 업데이트 (실제 API에서 제공하는 경우)
        setProgress(prev => Math.min(prev + 10, 90));
      }
    } catch (err) {
      setStatus('error');
      setError(err.message);
      setLoading(false);
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [selectedModel, selectedDataset, pollingInterval]);

  // Validation 실행
  const handleRunValidation = useCallback(async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Please select a model and dataset');
      return;
    }

    setStatus('running');
    setProgress(0);
    setError(null);
    setLoading(true);
    
    try {
      // Validation 시작
      const result = await startYoloValidation({
        uid,
        pid: selectedDataset.pid || 'P0001',
        tid: selectedDataset.tid || 'T0001',
        cid: 'yolo',
        did: selectedDataset.id,
        task_type: 'detection',
        parameters: validationParams
      });
      
      setCurrentVid(result.vid);
      
      // 폴링 시작
      const interval = setInterval(() => {
        pollValidationStatus(result.vid);
      }, 2000);
      
      setPollingInterval(interval);
      
    } catch (err) {
      setStatus('error');
      setError(err.message);
      setLoading(false);
    }
  }, [selectedModel, selectedDataset, validationParams, pollValidationStatus]);

  // Validation 파라미터 업데이트
  const updateValidationParams = useCallback((newParams) => {
    setValidationParams(prev => ({ ...prev, ...newParams }));
  }, []);

  // 컴포넌트 언마운트 시 폴링 정리
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  return {
    // 상태
    selectedModel,
    setSelectedModel: handleModelChange,
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
    validationParams,
    
    // 핸들러
    handleRunValidation,
    updateValidationParams,
    
    // 상수
    mockModels: MOCK_MODELS,
    deviceOptions: DEVICE_OPTIONS
  };
}; 