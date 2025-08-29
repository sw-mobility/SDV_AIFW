import { useState, useCallback } from 'react';
import { startYoloLabeling, DEFAULT_YOLO_PARAMS } from '../../api';

export const useLabelingWorkspace = (dataset) => {
  const [modelType, setModelType] = useState('YOLO');
  const [taskType, setTaskType] = useState('Object detection');
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [labelingParams, setLabelingParams] = useState({
    ...DEFAULT_YOLO_PARAMS,
    project: 'runs/labeling'
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // 라벨링 실행
  const handleRunLabeling = useCallback(async () => {
    if (!dataset) return;
    
    setStatus('running');
    setProgress(0);
    setError(null);
    setResult(null);
    
    try {
      const params = {
        pid: dataset.projectId || 'default',
        did: dataset.id,
        name: 'labeling_' + Date.now(),
        cid: 'default',
        parameters: {
          ...labelingParams,
          name: 'exp' // parameters 내부의 name은 기본값 사용
        }
      };

      const result = await startYoloLabeling(params);
      setResult(result);
      setStatus('success');
      setProgress(100);
    } catch (err) {
      setError(err.message);
      setStatus('error');
      setProgress(0);
    }
  }, [dataset, labelingParams]);

  // 모델 타입 변경
  const handleModelTypeChange = useCallback((newModelType) => {
    setModelType(newModelType);
  }, []);

  // 태스크 타입 변경
  const handleTaskTypeChange = useCallback((newTaskType) => {
    setTaskType(newTaskType);
  }, []);

  // 파라미터 변경
  const handleParamChange = useCallback((key, value) => {
    setLabelingParams(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);

  // 파라미터 초기화
  const resetParams = useCallback(() => {
    setLabelingParams({
      ...DEFAULT_YOLO_PARAMS,
      project: 'runs/labeling'
    });
  }, []);

  return {
    // 상태
    modelType,
    taskType,
    status,
    progress,
    labelingParams,
    result,
    error,
    
    // 핸들러
    handleRunLabeling,
    handleModelTypeChange,
    handleTaskTypeChange,
    handleParamChange,
    resetParams,
    
    // 유틸리티
    isDisabled: !dataset || status === 'running'
  };
}; 