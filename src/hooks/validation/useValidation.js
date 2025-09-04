import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { startYoloValidation, getValidationStatus, getValidationList } from '../../api/validation.js';
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
    tid: 'T0001', // Training ID - 기본값 설정
    model: 'best.pt',
    task_type: 'detection',
    imgsz: 640,
    batch: 32,
    device: 'gpu',
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
      const response = await fetchLabeledDatasets({ uid });
      console.log('Fetched labeled datasets:', response);
      
      if (response && response.data && Array.isArray(response.data)) {
        setDatasets(response.data);
      } else {
        console.warn('Invalid datasets response:', response);
        setDatasets([]);
      }
    } catch (err) {
      setDatasetError(err.message);
      console.error('Failed to fetch labeled datasets:', err);
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  // 컴포넌트 마운트 시 데이터셋 목록 가져오기
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // 선택된 데이터셋이 변경될 때 validation 파라미터에 dataset 관련 정보 설정
  useEffect(() => {
    if (selectedDataset) {
      // Dataset에서 추출할 수 있는 정보로 validation 파라미터 업데이트
      const modelPath = selectedDataset.tid ? `${selectedDataset.tid}/best.pt` : 'best.pt';
      const projectId = selectedDataset.pid || selectedDataset.projectId || 'P0001';
      
      setValidationParams(prev => ({
        ...prev,
        model: modelPath,
        pid: projectId
      }));
    }
  }, [selectedDataset]);

  // Validation History 새로고침 함수
  const refreshValidationHistory = useCallback(async () => {
    try {
      console.log('Refreshing validation history...');
      const validationList = await getValidationList({ uid });
      console.log('Refreshed validation list:', validationList);
      
      if (validationList && validationList.length > 0) {
        const latestResults = validationList.map(validation => ({
          vid: validation.vid,
          model: validation.parameters?.model || validation.used_codebase || 'Unknown',
          dataset: validation.dataset_name || '',
          timestamp: validation.created_at,
          status: validation.status,
          metrics: validation.metrics_summary || {},
          result_path: validation.artifacts_path,
          plots_path: null
        }));
        setResults(latestResults);
      }
      
      // ValidationHistoryList도 자동으로 refresh되도록 강제로 상태 변경
      // 이는 ValidationHistoryList가 useEffect의 의존성 배열에 uid를 포함하고 있기 때문
      console.log('Validation history refreshed successfully');
    } catch (error) {
      console.error('Failed to refresh validation history:', error);
    }
  }, [uid]);

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
        
        // Validation이 완료되면 ValidationHistoryList를 자동으로 refresh
        console.log('Validation completed! Triggering automatic refresh of ValidationHistoryList...');
        
        // Validation이 완료되면 refreshValidationHistory를 호출하여 ValidationHistoryList도 자동으로 refresh
        await refreshValidationHistory();
      } else if (result.status === 'failed' || result.status === 'error') {
        setStatus('error');
        setError(result.error || result.message || 'Validation failed');
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
        
        // Validation이 실패해도 ValidationHistoryList를 자동으로 refresh
        console.log('Validation failed! Triggering automatic refresh of ValidationHistoryList...');
        await refreshValidationHistory();
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
  }, [selectedDataset, pollingInterval, validationParams.model, refreshValidationHistory]);

  // Validation 실행
  const handleRunValidation = useCallback(async () => {
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }

    if (!validationParams.tid || validationParams.tid.trim() === '') {
      setError('Training ID는 필수 입력 항목입니다.');
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
      console.log('Dataset details:', {
        did: selectedDataset.did,
        _id: selectedDataset._id,
        id: selectedDataset.id,
        name: selectedDataset.name,
        type: selectedDataset.type,
        datasetType: selectedDataset.datasetType,
        pid: selectedDataset.pid,
        tid: selectedDataset.tid
      });
      
      // API 스펙에 맞는 요청 구조
      // Labeled dataset의 경우, 해당 dataset과 연결된 training ID를 사용
      const trainingId = selectedDataset.tid || selectedDataset.origin_tid || validationParams.tid || 'T0001';
      
      console.log('Training ID selection:', {
        selectedDatasetTid: selectedDataset.tid,
        selectedDatasetOriginTid: selectedDataset.origin_tid,
        validationParamsTid: validationParams.tid,
        finalTrainingId: trainingId
      });
      
      const requestData = {
        pid: selectedDataset.pid || 'P0001',
        tid: trainingId, // 데이터셋과 연결된 training ID 우선 사용
        cid: 'yolo',
        did: datasetId,
        task_type: validationParams.task_type,
        parameters: validationParams
      };

      console.log('Starting validation with:', requestData);
      console.log('Validation parameters:', validationParams);
      
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
      tid: 'T0001', // Training ID - 기본값 설정
      model: 'best.pt',
      task_type: 'detection',
      imgsz: 640,
      batch: 32,
      device: 'gpu',
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
    resetValidationParams,
    refreshValidationHistory
  };
}; 