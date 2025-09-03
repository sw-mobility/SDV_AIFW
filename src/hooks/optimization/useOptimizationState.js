import { useState, useCallback } from 'react';
import { runOptimization, getOptimizationList } from '../../api/optimization.js';
import { uid } from '../../api/uid.js';

/**
 * Optimization 페이지의 상태 관리 훅
 * API 명세서에 따라 구현
 */
const useOptimizationState = () => {
  const [optimizationType, setOptimizationType] = useState('');
  const [modelType, setModelType] = useState('training'); // 'training' 또는 'optimization'
  const [modelId, setModelId] = useState(''); // TID 또는 OID
  const [optimizationParams, setOptimizationParams] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  // results state 제거 - Optimization History로 통일
  const [error, setError] = useState(null); // 에러 상태 추가

  const handleOptimizationTypeChange = useCallback((type) => {
    setOptimizationType(type);
    setError(null); // 에러 초기화
    
    const baseParams = {
      model_id: '', // Empty by default, user must input
      model_name: 'best.pt' // Default model filename
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
        setOptimizationParams(baseParams); // Only model_id and model_name needed
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

  // Model type 변경 (training 또는 optimization)
  const handleModelTypeChange = useCallback((type) => {
    setModelType(type);
    setModelId(''); // 모델 타입 변경 시 ID 초기화
    setError(null);
  }, []);

  // Model ID 변경 (TID 또는 OID)
  const handleModelIdChange = useCallback((id) => {
    setModelId(id);
    setError(null);
  }, []);

  const handleRunOptimization = useCallback(async () => {
    if (!optimizationType) {
      console.error('Optimization type is required');
      return;
    }

    if (!modelId || modelId.trim() === '') {
      setError('Model ID는 필수 입력 항목입니다.');
      return;
    }

    setIsRunning(true);
    setStatus('running');
    setProgress(0);

    try {
      // PID는 기본값으로 P0001 사용 (다른 페이지들과 동일)
      const pid = 'P0001';
      const currentUid = uid; // uid.js에서 가져온 값 사용

      console.log('Optimization request params:', {
        optimizationType,
        modelType,
        modelId,
        params: optimizationParams,
        pid,
        uid: currentUid
      });

      setProgress(10);

      const response = await runOptimization(optimizationType, { ...optimizationParams, model_id: modelId }, pid, currentUid);
      
      setProgress(50);
      setProgress(80);

      setProgress(100);
      setStatus('success');
      
      // Optimization 완료 시 history 자동 새로고침
      setTimeout(() => {
        refreshOptimizationHistory();
      }, 1000);

    } catch (error) {
      console.error('Optimization failed:', error);
      setStatus('error');
      
      // 에러 메시지 개선
      let errorMessage = 'Unknown error occurred';
      
      if (error.message) {
        if (error.message.includes('404') || error.message.includes('Not Found')) {
          errorMessage = '존재하지 않는 Model ID이거나 잘못 입력한 값이 존재합니다.';
        } else if (error.message.includes('422') || error.message.includes('Validation Error')) {
          errorMessage = '입력한 파라미터 값이 올바르지 않습니다. 값을 확인해주세요.';
        } else if (error.message.includes('500') || error.message.includes('Internal Server Error')) {
          errorMessage = '입력하신 Model ID가 올바른지 확인해주세요. 존재하지 않는 Model ID일 수 있습니다.';
        } else {
          errorMessage = error.message;
        }
      }
      
      setError(errorMessage);
    } finally {
      setIsRunning(false);
    }
  }, [optimizationType, optimizationParams, runOptimization]);

  const resetOptimization = useCallback(() => {
    setOptimizationType('');
    setModelType('training');
    setModelId('');
    setOptimizationParams({});
    setIsRunning(false);
    setProgress(0);
    setStatus('idle');
    // results 초기화 제거 - Optimization History로 통일
    setError(null); // 에러 초기화
  }, []);

  // Optimization History 새로고침 함수
  const refreshOptimizationHistory = useCallback(async () => {
    try {
      // OptimizationHistoryList 컴포넌트에서 직접 API 호출하므로
      // 여기서는 단순히 콜백만 제공
      console.log('Optimization history refresh requested');
    } catch (error) {
      console.error('Failed to refresh optimization history:', error);
    }
  }, []);

  return {
    optimizationType,
    setOptimizationType,
    modelType,
    modelId,
    optimizationParams,
    setOptimizationParams,
    isRunning,
    progress,
    status,
    // results 제거 - Optimization History로 통일
    error,
    handleRunOptimization,
    handleOptimizationTypeChange,
    handleModelTypeChange,
    handleModelIdChange,
    handleParamChange,
    resetOptimization,
    refreshOptimizationHistory
  };
};

export default useOptimizationState; 