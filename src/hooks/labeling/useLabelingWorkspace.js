import { useState, useCallback } from 'react';

export const useLabelingWorkspace = (dataset) => {
  const [modelType, setModelType] = useState('YOLO');
  const [taskType, setTaskType] = useState('Object detection');
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);

  // 라벨링 실행
  const handleRunLabeling = useCallback(() => {
    if (!dataset) return;
    
    setStatus('running');
    setProgress(0);
    
    let pct = 0;
    const interval = setInterval(() => {
      pct += 10;
      setProgress(pct);
      if (pct >= 100) {
        clearInterval(interval);
        setStatus('success');
      }
    }, 400);
  }, [dataset]);

  // 모델 타입 변경
  const handleModelTypeChange = useCallback((newModelType) => {
    setModelType(newModelType);
  }, []);

  // 태스크 타입 변경
  const handleTaskTypeChange = useCallback((newTaskType) => {
    setTaskType(newTaskType);
  }, []);

  return {
    // 상태
    modelType,
    taskType,
    status,
    progress,
    
    // 핸들러
    handleRunLabeling,
    handleModelTypeChange,
    handleTaskTypeChange,
    
    // 유틸리티
    isDisabled: !dataset || status === 'running'
  };
}; 