import { useState, useCallback, useEffect } from 'react';
import { startYoloLabeling, DEFAULT_YOLO_PARAMS } from '../../api';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

export const useLabelingWorkspace = (dataset) => {
  const [modelType, setModelType] = useState('YOLO');
  const [taskType, setTaskType] = useState('Object detection');
  const [status, setStatus] = useState('idle'); // idle | running | success | error
  const [progress, setProgress] = useState(0);
  const [labelingParams, setLabelingParams] = useState({
    ...DEFAULT_YOLO_PARAMS,
    project: 'runs/labeling'
  });
  const [selectedParamKeys, setSelectedParamKeys] = useState(['model']); // 기본적으로 model은 선택됨
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [labeledDatasets, setLabeledDatasets] = useState([]);
  const [isPolling, setIsPolling] = useState(false);

  // 라벨링 실행
  const handleRunLabeling = useCallback(async () => {
    if (!dataset) return;
    
    setStatus('running');
    setProgress(0);
    setError(null);
    setResult(null);
    
    try {
      // API 스펙에 맞춰 모든 필수 파라미터 포함
      const parameters = {
        model: labelingParams.model || 'yolo11n',
        conf: labelingParams.conf || 0.25,
        iou: labelingParams.iou || 0.45,
        imgsz: labelingParams.imgsz || 640,
        rect: labelingParams.rect || false,
        half: labelingParams.half || false,
        device: labelingParams.device || 'cpu',
        batch: labelingParams.batch || 1,
        max_det: labelingParams.max_det || 300,
        vid_stride: labelingParams.vid_stride || 1,
        stream_buffer: labelingParams.stream_buffer || false,
        visualize: labelingParams.visualize || false,
        augment: labelingParams.augment || false,
        agnostic_nms: labelingParams.agnostic_nms || false,
        classes: labelingParams.classes || [0],
        retina_masks: labelingParams.retina_masks || false,
        embed: labelingParams.embed || [0],
        project: labelingParams.project || 'runs/detect',
        name: labelingParams.name || 'exp',
        stream: labelingParams.stream || false,
        verbose: labelingParams.verbose || false,
        show: labelingParams.show || false,
        save: labelingParams.save || true,
        save_frames: labelingParams.save_frames || false,
        save_txt: labelingParams.save_txt || true,
        save_conf: labelingParams.save_conf || false,
        save_crop: labelingParams.save_crop || false,
        show_labels: labelingParams.show_labels || true,
        show_conf: labelingParams.show_conf || true,
        show_boxes: labelingParams.show_boxes || true,
        line_width: labelingParams.line_width || 3
      };

      console.log('Dataset object:', dataset); // 디버깅용

      const params = {
        pid: dataset?.projectId || 'P0001',
        did: dataset?.id || 'R0001',
        name: dataset?.name || 'l',
        parameters: parameters
      };

      console.log('API Request:', params); // 디버깅용

      const result = await startYoloLabeling(params);
      
      // API가 성공적으로 호출되면 폴링 시작
      if (result) {
        setIsPolling(true);
        setStatus('running');
        setProgress(50); // API 호출 완료, 이제 폴링 중
        setResult('Labeling started successfully. Waiting for completion...');
      }
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

  // Labeled datasets 조회
  const fetchLabeledDatasetsList = useCallback(async () => {
    try {
      const res = await fetchLabeledDatasets({ uid });
      setLabeledDatasets(res.data || []);
    } catch (err) {
      console.error('Failed to fetch labeled datasets:', err);
    }
  }, []);

  // 초기 로드
  useEffect(() => {
    fetchLabeledDatasetsList();
  }, [fetchLabeledDatasetsList]);

  // Labeled datasets 폴링 (라벨링 완료 확인용)
  useEffect(() => {
    if (!isPolling) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetchLabeledDatasets({ uid });
        const newDatasets = res.data || [];
        
        // 새로운 labeled dataset이 추가되었는지 확인
        if (newDatasets.length > labeledDatasets.length) {
          setLabeledDatasets(newDatasets);
          setIsPolling(false);
          setStatus('success');
          setProgress(100);
          setResult('Labeling completed! New labeled dataset has been created.');
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 5000); // 5초마다 체크

    return () => clearInterval(interval);
  }, [isPolling, labeledDatasets.length]);

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
    selectedParamKeys,
    result,
    error,
    labeledDatasets,
    isPolling,
    
    // 핸들러
    handleRunLabeling,
    handleModelTypeChange,
    handleTaskTypeChange,
    handleParamChange,
    resetParams,
    setSelectedParamKeys,
    fetchLabeledDatasetsList,
    
    // 유틸리티
    isDisabled: !dataset || status === 'running' || isPolling
  };
}; 