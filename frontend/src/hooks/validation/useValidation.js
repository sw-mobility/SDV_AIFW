import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { startYoloValidation, getValidationStatus, getValidationList } from '../../api/validation.js';
import { fetchCodebases, fetchCodebase } from '../../api/codeTemplates.js';
import { uid } from '../../api/uid.js';

export const useValidation = (projectId) => {
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

  // Codebase 관련 상태 (training과 동일하게 직접 관리)
  const [codebases, setCodebases] = useState([]);
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [codebaseLoading, setCodebaseLoading] = useState(false);
  const [codebaseError, setCodebaseError] = useState(null);
  const [codebaseFileStructure, setCodebaseFileStructure] = useState([]);
  const [codebaseFiles, setCodebaseFiles] = useState({});
  const [codebaseFilesLoading, setCodebaseFilesLoading] = useState(false);
  const [showCodeEditor, setShowCodeEditor] = useState(false);

  // Codebase 관련 함수들 (training과 동일)
  const fetchCodebasesList = useCallback(async () => {
    setCodebaseLoading(true);
    setCodebaseError(null);
    try {
      const data = await fetchCodebases();
      const sortedData = data.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0);
        const dateB = new Date(b.updated_at || b.created_at || 0);
        return dateB - dateA;
      });
      setCodebases(sortedData);
    } catch (err) {
      setCodebaseError(err.message);
      console.error('Failed to load codebases:', err);
    } finally {
      setCodebaseLoading(false);
    }
  }, []);

  const fetchCodebaseFiles = useCallback(async (codebaseId) => {
    setCodebaseFilesLoading(true);
    try {
      const codebaseData = await fetchCodebase(codebaseId);
      
      setCodebaseFileStructure(codebaseData.tree || []);
      setCodebaseFiles(codebaseData.files || {});
    } catch (err) {
      console.error('Failed to load codebase files:', err);
      setCodebaseFileStructure([]);
      setCodebaseFiles({});
    } finally {
      setCodebaseFilesLoading(false);
    }
  }, []);

  // selectedCodebase가 변경될 때 파일 정보 로드
  useEffect(() => {
    if (selectedCodebase?.cid) {
      fetchCodebaseFiles(selectedCodebase.cid);
    } else {
      setCodebaseFileStructure([]);
      setCodebaseFiles({});
    }
  }, [selectedCodebase, fetchCodebaseFiles]);

  // 초기 codebase 목록 로드
  useEffect(() => {
    fetchCodebasesList();
  }, [fetchCodebasesList]);

  // 데이터셋 목록 조회
  const fetchDatasets = useCallback(async () => {
    setDatasetLoading(true);
    setDatasetError(null);
    try {
      const response = await fetchLabeledDatasets({ uid });
      console.log('Fetched labeled datasets:', response);
      
      if (response && response.data && Array.isArray(response.data)) {
        // 모든 labeled dataset을 표시 (training과 동일하게)
        const formattedDatasets = response.data.map(ds => ({
          ...ds, // Keep all original fields including did, _id, etc.
          id: ds.did || ds._id, // Add id field for backward compatibility
          size: ds.total,
          labelCount: ds.total,
        }));
        setDatasets(formattedDatasets);
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
  }, [uid]);

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

  // Validation 완료 시 자동 refresh 함수 (ValidationHistoryList의 REFRESH 버튼 클릭과 동일)
  const triggerResultsRefresh = useCallback(() => {
    console.log('Validation completed! Triggering results refresh...');
    // ValidationHistoryList의 handleRefresh와 동일한 동작
    // onRefresh prop이 있으면 호출하고, 없으면 자체 fetchValidations 실행
    if (refreshValidationHistory) {
      refreshValidationHistory();
    }
  }, [refreshValidationHistory]);

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
        
        // Validation이 완료되면 RESULTS의 REFRESH 버튼 클릭과 동일한 동작
        console.log('Validation completed! Triggering results refresh...');
        triggerResultsRefresh();
      } else if (result.status === 'failed' || result.status === 'error') {
        setStatus('error');
        setError(result.error || result.message || 'Validation failed');
        setLoading(false);
        clearInterval(pollingInterval);
        setPollingInterval(null);
        
        // Validation이 실패해도 RESULTS의 REFRESH 버튼 클릭과 동일한 동작
        console.log('Validation failed! Triggering results refresh...');
        triggerResultsRefresh();
      } else if (result.status === 'running') {
        // 진행률 업데이트 (실제 API에서 제공하는 경우)
        if (result.progress !== undefined) {
          setProgress(result.progress);
        } else {
          setProgress(prev => Math.min(prev + 10, 90));
        }
      }
    } catch (err) {
      console.error('Polling error for vid:', vid, err);
      
      // 500 에러나 서버 오류의 경우 polling을 계속하되, 로그만 추가
      if (err.message.includes('500') || err.message.includes('Internal Server Error')) {
        console.warn('Server error during polling, continuing to poll...');
        // polling을 중단하지 않고 계속 시도
        return;
      }
      
      // 다른 오류의 경우에만 polling 중단
      setStatus('error');
      setError(`Status check failed: ${err.message}`);
      setLoading(false);
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [selectedDataset, pollingInterval, validationParams.model, triggerResultsRefresh]);

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
        projectId: selectedDataset.projectId,
        tid: selectedDataset.tid,
        allFields: Object.keys(selectedDataset) // 모든 필드 확인
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
      
      // device 파라미터를 변환 (gpu: cuda, cpu: cpu 그대로)
      const convertedParams = {
        ...validationParams,
        device: validationParams.device === 'gpu' ? 'cuda' : validationParams.device
      };

      // pid 결정: 데이터셋의 pid > projectId > 현재 페이지의 projectId > 기본값
      const datasetPid = selectedDataset.pid || selectedDataset.projectId;
      const finalPid = datasetPid || projectId || 'P0001';
      
      console.log('PID selection:', {
        datasetPid: selectedDataset.pid,
        datasetProjectId: selectedDataset.projectId,
        currentProjectId: projectId,
        finalPid: finalPid
      });
      
      const requestData = {
        pid: finalPid,
        tid: trainingId, // 데이터셋과 연결된 training ID 우선 사용
        cid: 'yolo',
        did: datasetId,
        task_type: validationParams.task_type,
        parameters: convertedParams
      };

      console.log('Starting validation with:', requestData);
      console.log('Original validation parameters:', validationParams);
      console.log('Converted parameters (device as number):', convertedParams);
      
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
      
      // 첫 번째 폴링은 3초 후에 시작 (서버 초기화 시간 확보)
      setTimeout(() => {
        pollValidationStatus(vid);
        
        // 그 다음부터는 5초마다 폴링
        const interval = setInterval(() => {
          pollValidationStatus(vid);
        }, 5000);
        
        setPollingInterval(interval);
      }, 3000);
      
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
    
    // Codebase 상태 (training과 동일)
    codebases,
    selectedCodebase,
    setSelectedCodebase,
    codebaseLoading,
    codebaseError,
    codebaseFileStructure,
    codebaseFiles,
    codebaseFilesLoading,
    showCodeEditor,
    setShowCodeEditor,
    
    // 핸들러
    handleRunValidation,
    updateValidationParams,
    resetValidationParams,
    refreshValidationHistory
  };
}; 