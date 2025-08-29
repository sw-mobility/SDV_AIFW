import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { startYoloValidation, getValidationStatus } from '../../api/validation.js';
import { uid } from '../../api/uid.js';

export const useValidation = () => {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // YOLO validation parameters (API 스펙에 맞게 구성)
  const [validationParams, setValidationParams] = useState({
    model: 'best.pt',
    task_type: 'detection',
    imgsz: 640,
    batch: 32,
    device: 'cpu',
    workers: 8,
    conf: 0.001,
    iou: 0.6,
    max_det: 300,
    save_json: true,
    save_txt: true,
    save_conf: true,
    plots: true,
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
      console.log('Fetched datasets:', res.data);
      if (res.data && res.data.length > 0) {
        console.log('First dataset structure:', res.data[0]);
        console.log('Available fields:', Object.keys(res.data[0]));
      }
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

  // Dataset 선택 시 모델 자동 설정
  useEffect(() => {
    if (selectedDataset) {
      console.log('Dataset selected:', selectedDataset);
      console.log('Dataset fields:', Object.keys(selectedDataset));
      
      // Dataset에서 모델 정보를 가져와서 설정
      // 예: dataset에 model_path나 model_name 필드가 있다면 사용
      const modelPath = selectedDataset.model_path || selectedDataset.model_name || 'best.pt';
      setValidationParams(prev => ({
        ...prev,
        model: modelPath
      }));
    }
  }, [selectedDataset]);

  // Validation 상태 폴링
  const pollValidationStatus = useCallback(async (vid) => {
    try {
      const result = await getValidationStatus({ vid });
      
      console.log('Polling validation status:', result);
      
      // API 응답에 따라 상태 업데이트
      if (result.status === 'completed') {
        setStatus('success');
        setProgress(100);
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
        
        // 결과 추가 (metrics 정보 포함)
        setResults(prev => [
          ...prev,
          {
            vid: result.vid || vid,
            model: validationParams.model,
            dataset: selectedDataset?.name || '',
            timestamp: new Date().toISOString(),
            status: result.status,
            metrics: result.metrics || {},
            result_path: result.result_path,
            plots_path: result.plots_path
          }
        ]);
      } else if (result.status === 'failed' || result.status === 'error') {
        setStatus('error');
        setError(result.error || result.message || 'Validation failed');
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
      } else if (result.status === 'running') {
        // 진행률 업데이트 (실제 API에서 제공하는 경우)
        if (result.progress !== undefined) {
          setProgress(result.progress);
        } else {
          setProgress(prev => Math.min(prev + 10, 90));
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
      setStatus('error');
      setError(err.message);
      setLoading(false);
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [selectedDataset, pollingInterval, validationParams.model]);

  // Validation 실행
  const handleRunValidation = useCallback(async () => {
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }

    setStatus('running');
    setProgress(0);
    setError(null);
    setLoading(true);
    
    try {
      // dataset ID 추출 (did 필드만 사용)
      const datasetId = selectedDataset.did;
      
      if (!datasetId) {
        throw new Error('Dataset ID not found. Please select a valid dataset.');
      }
      
      console.log('Selected dataset:', selectedDataset);
      console.log('Dataset ID:', datasetId);
      
      // API 스펙에 맞는 요청 구조
      const requestData = {
        pid: selectedDataset.pid || 'P0001',
        tid: selectedDataset.tid || 'T0001', 
        cid: 'yolo',
        did: datasetId,
        task_type: validationParams.task_type,
        parameters: validationParams
      };

      console.log('Starting validation with:', requestData);
      
      // Validation 시작
      const result = await startYoloValidation({
        ...requestData,
        uid
      });
      
      console.log('Validation started:', result);
      
      // vid 추출 (API 응답에서 vid 필드 사용)
      const vid = result.vid;
      if (!vid) {
        throw new Error('No validation ID received from server');
      }
      
      setCurrentVid(vid);
      
      // 폴링 시작 (5초마다)
      const interval = setInterval(() => {
        pollValidationStatus(vid);
      }, 5000);
      
      setPollingInterval(interval);
      
    } catch (err) {
      console.error('Validation start error:', err);
      setStatus('error');
      setError(err.message);
      setLoading(false);
    }
  }, [selectedDataset, validationParams, pollValidationStatus]);

  // Validation 파라미터 업데이트
  const updateValidationParams = useCallback((newParams) => {
    setValidationParams(prev => ({ ...prev, ...newParams }));
  }, []);

  // Validation 파라미터 리셋
  const resetValidationParams = useCallback(() => {
    setValidationParams({
      model: 'best.pt',
      task_type: 'detection',
      imgsz: 640,
      batch: 32,
      device: 'cpu',
      workers: 8,
      conf: 0.001,
      iou: 0.6,
      max_det: 300,
      save_json: true,
      save_txt: true,
      save_conf: true,
      plots: true,
      verbose: true,
      half: false,
      dnn: false,
      agnostic_nms: false,
      augment: false,
      rect: false
    });
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
    resetValidationParams
  };
}; 