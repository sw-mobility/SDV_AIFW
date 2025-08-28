import { useState, useCallback } from 'react';
import { runOptimization } from '../../api/optimization.js';

/**
 * Optimization 페이지의 상태 관리 훅
 * Training/Validation 페이지와 동일한 패턴으로 구현
 */
const useOptimizationState = () => {
  // Core state
  const [selectedModel, setSelectedModel] = useState('');
  const [optimizationType, setOptimizationType] = useState('');
  const [optimizationParams, setOptimizationParams] = useState({});
  
  // Execution state
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle'); // 'idle', 'running', 'success', 'error'
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);

  // Event handlers
  const handleModelChange = useCallback((model) => {
    setSelectedModel(model);
    // 모델 정보를 파라미터에 저장 (API 요청 시 사용)
    const modelInfo = {
      model_name: model.split('/').pop() || 'yolov8n.pt',
      training_id: model.includes('T') ? model.split('/').find(part => part.startsWith('T')) : 'T0001'
    };
    setOptimizationParams(prev => ({
      ...prev,
      ...modelInfo
    }));
  }, []);

  const handleOptimizationTypeChange = useCallback((type) => {
    setOptimizationType(type);
    // 최적화 타입이 변경되면 파라미터 초기화 (모델 정보는 유지)
    const baseParams = {
      training_id: 'T0001',
      model_name: 'yolov8n.pt'
    };

    // 최적화 타입별 기본 파라미터 설정
    switch (type) {
      case 'pt_to_onnx_fp32':
      case 'pt_to_onnx_fp16':
        setOptimizationParams({
          ...baseParams,
          input_size: [640, 640],
          batch_size: 1,
          opset_version: 11,
          dynamic_axes: false
        });
        break;
      case 'prune_unstructured':
        setOptimizationParams({
          ...baseParams,
          amount: 0.2,
          pruning_type: 'l1_unstructured',
          global_unstructured: false,
          importance: 'magnitude'
        });
        break;
      case 'prune_structured':
        setOptimizationParams({
          ...baseParams,
          amount: 0.2,
          pruning_type: 'ln_structured',
          n: 2,
          dim: 0
        });
        break;
      case 'check_model_stats':
        setOptimizationParams({
          ...baseParams,
          save_stats: true,
          detailed_stats: false,
          save_format: 'json'
        });
        break;
      default:
        setOptimizationParams(baseParams);
    }
  }, [selectedModel]);

  const handleParamChange = useCallback((key, value, param) => {
    setOptimizationParams(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);

  const handleRunOptimization = useCallback(async () => {
    if (!selectedModel || !optimizationType) {
      console.error('Model and optimization type are required');
      return;
    }

    setIsRunning(true);
    setStatus('running');
    setProgress(0);
    setLogs([]);
    setResults(null);

    try {
      // API 호출을 위한 기본 파라미터 설정
      const pid = 'P0001'; // Project ID
      const oid = 'O0001'; // Optimization ID
      const uid = '0001';  // User ID

      // API 요청에 필요한 파라미터만 전달
      const params = {
        ...optimizationParams
      };

      setLogs(prev => [...prev, 'Initializing optimization...']);
      setProgress(10);

      // 실제 API 호출
      const response = await runOptimization(optimizationType, params, pid, oid, uid);
      
      setLogs(prev => [...prev, 'Optimization request sent successfully']);
      setProgress(50);
      
      // API 응답 처리
      setLogs(prev => [...prev, 'Processing response...']);
      setProgress(80);

      // 성공 결과 설정
      const getOutputPath = () => {
        const basePath = `artifacts/${pid}/optimizing/${oid}`;
        switch (optimizationType) {
          case 'pt_to_onnx_fp32':
            return `${basePath}/model_fp32.onnx`;
          case 'pt_to_onnx_fp16':
            return `${basePath}/model_fp16.onnx`;
          case 'prune_unstructured':
          case 'prune_structured':
            return `${basePath}/pruned_model.pt`;
          case 'check_model_stats':
            return `${basePath}/model_stats.json`;
          default:
            return `${basePath}/optimized_model`;
        }
      };

      setResults({
        outputPath: getOutputPath(),
        statsPath: optimizationType === 'check_model_stats' ? getOutputPath() : null,
        processingTime: 'API response received',
        taskId: response // API에서 반환된 task ID
      });

      setLogs(prev => [...prev, 'Optimization completed successfully!']);
      setProgress(100);
      setStatus('success');

    } catch (error) {
      console.error('Optimization failed:', error);
      setStatus('error');
      setLogs(prev => [...prev, `Error: ${error.message}`]);
    } finally {
      setIsRunning(false);
    }
  }, [selectedModel, optimizationType, optimizationParams]);

  return {
    // Core state
    selectedModel,
    setSelectedModel,
    optimizationType,
    setOptimizationType,
    optimizationParams,
    setOptimizationParams,
    
    // Execution state
    isRunning,
    progress,
    status,
    logs,
    results,
    
    // Event handlers
    handleRunOptimization,
    handleModelChange,
    handleOptimizationTypeChange,
    handleParamChange
  };
};

export default useOptimizationState; 