import { useState, useCallback } from 'react';
import { runOptimization } from '../../api/optimization.js';
import { uid } from '../../api/uid.js';

/**
 * Optimization 페이지의 상태 관리 훅
 * API 명세서에 따라 구현
 */
const useOptimizationState = () => {
  const [optimizationType, setOptimizationType] = useState('');
  const [optimizationParams, setOptimizationParams] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [results, setResults] = useState([]); // 배열로 변경

  const handleOptimizationTypeChange = useCallback((type) => {
    setOptimizationType(type);
    
    const baseParams = {
      training_id: 'T0001', // Changed to T0001
      model_name: 'best.pt' // Fixed to best.pt
    };
    
    setOptimizationParams(baseParams); // Set base params immediately

    switch (type) {
      case 'pt_to_onnx_fp32':
      case 'pt_to_onnx_fp16':
        setOptimizationParams(prev => ({
          ...prev,
          input_size: [640, 640],
          batch_size: 1,
          channels: 3
        }));
        break;
      case 'onnx_to_trt':
        setOptimizationParams(prev => ({
          ...prev,
          precision: 'fp32',
          device: 'gpu'
        }));
        break;
      case 'onnx_to_trt_int8':
        setOptimizationParams(prev => ({
          ...prev,
          calib_dir: '/app/int8_calib_images',
          device: 'gpu',
          mixed_fp16: false,
          sparse: false,
          int8_max_batches: 10,
          input_size: [640, 640],
          workspace_mib: 2048
        }));
        break;
      case 'prune_unstructured':
        setOptimizationParams(prev => ({
          ...prev,
          amount: 0.2,
          pruning_type: 'l1_unstructured'
        }));
        break;
      case 'prune_structured':
        setOptimizationParams(prev => ({
          ...prev,
          amount: 0.2,
          pruning_type: 'ln_structured',
          n: 2,
          dim: 0
        }));
        break;
      case 'check_model_stats':
        setOptimizationParams(baseParams); // Only input_path needed
        break;
      default:
        setOptimizationParams(baseParams);
    }
  }, []);

  const handleParamChange = useCallback((key, value) => {
    setOptimizationParams(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);

  const handleRunOptimization = useCallback(async () => {
    if (!optimizationType) {
      console.error('Optimization type is required');
      return;
    }

    setIsRunning(true);
    setStatus('running');
    setProgress(0);

    try {
      const pid = 'P0001'; // TODO: 실제 프로젝트 ID를 동적으로 가져와야 함
      const currentUid = uid; // uid.js에서 가져온 값 사용

      console.log('Optimization request params:', {
        optimizationType,
        params: optimizationParams,
        pid,
        uid: currentUid
      });

      setProgress(10);

      const response = await runOptimization(optimizationType, optimizationParams, pid, currentUid);
      
      setProgress(50);
      setProgress(80);

      // 새로운 결과를 기존 배열에 추가
      setResults(prevResults => [...prevResults, response]);

      setProgress(100);
      setStatus('success');

    } catch (error) {
      console.error('Optimization failed:', error);
      setStatus('error');
      const errorMessage = error.message || 'Unknown error occurred';
    } finally {
      setIsRunning(false);
    }
  }, [optimizationType, optimizationParams, runOptimization]);

  const resetOptimization = useCallback(() => {
    setOptimizationType('');
    setOptimizationParams({});
    setIsRunning(false);
    setProgress(0);
    setStatus('idle');
    setResults([]); // 배열 초기화
  }, []);

  return {
    optimizationType,
    setOptimizationType,
    optimizationParams,
    setOptimizationParams,
    isRunning,
    progress,
    status,
    results,
    handleRunOptimization,
    handleOptimizationTypeChange,
    handleParamChange,
    resetOptimization
  };
};

export default useOptimizationState; 