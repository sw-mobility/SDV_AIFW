import { useState, useCallback, useRef, useEffect } from 'react';
import { runOptimization, getOptimizationList } from '../../api/optimization.js';
import { uid } from '../../api/uid.js';

/**
 * Optimization 페이지의 상태 관리 훅
 * API 명세서에 따라 구현
 */
const useOptimizationState = (projectId = 'P0001') => {
  const [optimizationType, setOptimizationType] = useState('');
  const [modelType, setModelType] = useState('training'); // 'training' 또는 'optimization'
  const [modelId, setModelId] = useState(''); // TID 또는 OID
  const [optimizationParams, setOptimizationParams] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  // results state 제거 - Optimization History로 통일
  const [error, setError] = useState(null); // 에러 상태 추가
  
  // 최신 modelId를 추적하기 위한 ref
  const modelIdRef = useRef(modelId);
  
  // modelId가 변경될 때마다 ref 업데이트
  useEffect(() => {
    modelIdRef.current = modelId;
  }, [modelId]);

  const handleOptimizationTypeChange = useCallback((type) => {
    setOptimizationType(type);
    setError(null); // 에러 초기화
    
    // 최적화 타입에 따라 가장 적합한 모델 타입 자동 선택
    let defaultModelType = 'training'; // 기본값
    
    // ONNX 관련 최적화는 optimization 모델을 기본으로 설정
    if (type === 'onnx_to_trt' || type === 'onnx_to_trt_int8') {
      defaultModelType = 'optimization';
    }
    
    setModelType(defaultModelType);
    setModelId(''); // 모델 ID 초기화
    
    const baseParams = {
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
    setOptimizationParams(prev => {
      const newParams = {
        ...prev,
        [key]: value
      };
      
      // precision이 fp32로 변경되면 device를 gpu로 자동 설정
      if (key === 'precision' && value === 'fp32' && prev.device && prev.device !== 'gpu') {
        newParams.device = 'gpu';
      }
      
      return newParams;
    });
  }, []);

  // Model type 변경 (training 또는 optimization)
  const handleModelTypeChange = useCallback((type) => {
    setModelType(type);
    setModelId(''); // 모델 타입 변경 시 ID 초기화
    setError(null);
  }, []);

  // Model ID 변경 (TID 또는 OID)
  const handleModelIdChange = useCallback((id) => {
    console.log('handleModelIdChange called with:', {
      id,
      idType: typeof id,
      idLength: id ? id.length : 0,
      idTrimmed: id ? id.trim() : '',
      isEmpty: !id || id.trim() === ''
    });
    setModelId(id);
    setError(null);
  }, []);

  const handleRunOptimization = useCallback(async () => {
    // ref를 사용하여 최신 modelId 가져오기
    const currentModelId = modelIdRef.current;
    
    console.log('handleRunOptimization called with:', {
      optimizationType,
      modelType,
      modelId: currentModelId,
      modelIdType: typeof currentModelId,
      modelIdLength: currentModelId ? currentModelId.length : 0,
      modelIdTrimmed: currentModelId ? currentModelId.trim() : '',
      isEmpty: !currentModelId || currentModelId.trim() === ''
    });

    if (!optimizationType) {
      console.error('Optimization type is required');
      setError('Optimization 타입을 선택해주세요.');
      return;
    }

    if (!currentModelId || currentModelId.trim() === '') {
      console.error('Model ID validation failed:', {
        modelId: currentModelId,
        modelIdType: typeof currentModelId,
        modelIdLength: currentModelId ? currentModelId.length : 0,
        modelIdTrimmed: currentModelId ? currentModelId.trim() : ''
      });
      setError('Model ID는 필수 입력 항목입니다.');
      return;
    }

    // Model ID 형식 검증 추가
    const trimmedModelId = currentModelId.trim();
    if (modelType === 'training' && !trimmedModelId.startsWith('T')) {
      setError('Training Model ID는 T로 시작해야 합니다. (예: T0001)');
      return;
    }
    if (modelType === 'optimization' && !trimmedModelId.startsWith('O')) {
      setError('Optimization Model ID는 O로 시작해야 합니다. (예: O0001)');
      return;
    }

    setIsRunning(true);
    setStatus('running');
    setProgress(0);

    try {
      // 현재 프로젝트의 PID 사용
      const pid = projectId;
      const currentUid = uid; // uid.js에서 가져온 값 사용

      console.log('Optimization request params:', {
        optimizationType,
        modelType,
        modelId,
        params: optimizationParams,
        pid,
        uid: currentUid
      });
      
      console.log('Model ID check:', {
        modelId,
        modelIdType: typeof modelId,
        modelIdLength: modelId ? modelId.length : 0,
        modelIdTrimmed: modelId ? modelId.trim() : ''
      });

      setProgress(10);

      console.log('Calling runOptimization with uid:', currentUid);
      const response = await runOptimization(optimizationType, { ...optimizationParams, model_id: currentModelId }, pid, currentUid);
      
      setProgress(50);
      
      // API가 성공적으로 호출되면 완료
      if (response && response.oid) {
        setProgress(100);
        setStatus('success');
        console.log('Optimization started successfully. OID:', response.oid);
        
        // 1회 history와 model list 새로고침
        setTimeout(() => {
          refreshOptimizationHistory();
          refreshModelList();
        }, 1000);
        
        // 추가로 3초 후에도 한 번 더 새로고침 (optimization이 완료될 때까지 기다림)
        setTimeout(() => {
          refreshOptimizationHistory();
          refreshModelList();
        }, 3000);
      } else {
        setProgress(100);
        setStatus('success');
        
        // 즉시 완료된 경우 history와 model list 새로고침
        setTimeout(() => {
          refreshOptimizationHistory();
          refreshModelList();
        }, 1000);
        
        // 추가로 3초 후에도 한 번 더 새로고침 (optimization이 완료될 때까지 기다림)
        setTimeout(() => {
          refreshOptimizationHistory();
          refreshModelList();
        }, 3000);
      }

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
          errorMessage = '서버 오류가 발생했습니다. 모델 리스트가 완전히 로드되었는지 확인하고 다시 시도해주세요.';
        } else if (error.message.includes('Failed to get optimization format')) {
          errorMessage = '모델 정보를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
        } else {
          errorMessage = error.message;
        }
      }
      
      setError(errorMessage);
    } finally {
      setIsRunning(false);
    }
  }, [optimizationType, modelType, optimizationParams, runOptimization]);

  // Optimization History 새로고침 함수
  const refreshOptimizationHistory = useCallback(async () => {
    try {
      console.log('Optimization history refresh requested');
      // OptimizationHistoryList가 자동으로 새로고침되도록 강제로 상태 변경
      // 이는 OptimizationHistoryList가 useEffect의 의존성 배열에 uid를 포함하고 있기 때문
      console.log('Optimization history refreshed successfully');
    } catch (error) {
      console.error('Failed to refresh optimization history:', error);
    }
  }, []);

  // Model list 새로고침을 위한 ref
  const refreshModelListRef = useRef(null);

  // Model list 새로고침 함수
  const refreshModelList = useCallback(() => {
    if (refreshModelListRef.current) {
      refreshModelListRef.current();
    }
  }, []);


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

  // Model list 새로고침 콜백 설정 함수
  const setRefreshModelListCallback = useCallback((callback) => {
    refreshModelListRef.current = callback;
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
    refreshOptimizationHistory,
    refreshModelList,
    setRefreshModelListCallback
  };
};

export default useOptimizationState; 