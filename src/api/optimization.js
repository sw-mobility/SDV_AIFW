import { API_BASE_URL } from './init.js';

/**
 * Optimization API
 * API 명세서에 따른 최적화 엔드포인트들
 */

// 공통 헤더 설정
const getHeaders = (uid = '0001') => ({
  'Content-Type': 'application/json',
  'uid': uid
});

/**
 * 공통 API 호출 함수
 */
const callOptimizationAPI = async (endpoint, data, uid = '0001') => {
  try {
    console.log(`${endpoint} request data:`, data);
    
    const response = await fetch(`${API_BASE_URL}/optimizing/${endpoint}`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    console.log(`${endpoint} response status:`, response.status);

    if (response.status === 202) {
      const result = await response.json();
      console.log(`${endpoint} success response:`, result);
      return result;
    } else if (response.status === 422) {
      const error = await response.json();
      console.error(`${endpoint} validation error:`, error);
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorBody = await response.text();
        if (errorBody) {
          errorMessage += ` - ${errorBody}`;
        }
      } catch (e) {
        // 에러 본문을 읽을 수 없는 경우 무시
      }
      console.error(`${endpoint} HTTP error:`, errorMessage);
      throw new Error(errorMessage);
    }
  } catch (error) {
    console.error(`${endpoint} failed:`, error);
    throw error;
  }
};

/**
 * PT to ONNX FP32 변환
 */
export const convertPtToOnnxFp32 = async (data, uid = '0001') => {
  return callOptimizationAPI('pt_to_onnx_fp32', data, uid);
};

/**
 * PT to ONNX FP16 변환
 */
export const convertPtToOnnxFp16 = async (data, uid = '0001') => {
  return callOptimizationAPI('pt_to_onnx_fp16', data, uid);
};

/**
 * 비정형 프루닝
 */
export const pruneUnstructured = async (data, uid = '0001') => {
  return callOptimizationAPI('prune_unstructured', data, uid);
};

/**
 * 정형 프루닝
 */
export const pruneStructured = async (data, uid = '0001') => {
  return callOptimizationAPI('prune_structured', data, uid);
};

/**
 * 모델 통계 확인
 */
export const checkModelStats = async (data, uid = '0001') => {
  return callOptimizationAPI('check_model_stats', data, uid);
};

/**
 * ONNX to TensorRT FP32/FP16 변환
 */
export const convertOnnxToTrt = async (data, uid = '0001') => {
  return callOptimizationAPI('onnx_to_trt', data, uid);
};

/**
 * ONNX to TensorRT INT8 변환 (엔트로피 캘리브레이션)
 */
export const convertOnnxToTrtInt8 = async (data, uid = '0001') => {
  return callOptimizationAPI('onnx_to_trt_int8', data, uid);
};

/**
 * 최적화 요청 데이터 생성 헬퍼 함수
 * Creates optimization request data according to API specification
 */
export const createOptimizationRequest = (optimizationType, params, pid, uid = '0001') => {
  // Helper function to create input_path based on model ID type
  const createInputPath = (modelId, modelName) => {
    if (modelId.startsWith('T')) {
      // Use training path for Training ID (best.pt)
      return `artifacts/${pid}/training/${modelId}/best.pt`;
    } else if (modelId.startsWith('O')) {
      // Use optimizing path for Optimizing ID (best.onnx)
      return `artifacts/${pid}/optimizing/${modelId}/best.onnx`;
    } else {
      // Default to training path for unknown ID format
      return `artifacts/${pid}/training/${modelId}/best.pt`;
    }
  };

  // Base request structure (according to API specification)
  const baseRequest = {
    pid,
    parameters: {
      kind: optimizationType,
      input_path: createInputPath(
        params.model_id || 'T0001'
      )
    }
  };

  // Helper function to get optimization-specific parameters
  const getOptimizationParams = (type, params) => {
    const baseParams = { ...baseRequest.parameters };
    
    switch (type) {
      case 'pt_to_onnx_fp32':
      case 'pt_to_onnx_fp16':
        return {
          ...baseParams,
          kind: 'pt_to_onnx',
          output_path: params.output_path || `best_${type.includes('fp16') ? 'fp16' : 'fp32'}.onnx`,
          input_size: params.input_size || [640, 640],
          batch_size: params.batch_size || 1,
          channels: params.channels || 3
        };
        
      case 'onnx_to_trt':
        return {
          ...baseParams,
          kind: 'onnx_to_trt',
          output_path: params.output_path || `best_${params.precision || 'fp32'}.engine`,
          precision: params.precision || 'fp32',
          device: params.device || 'gpu'
        };
        
      case 'onnx_to_trt_int8':
        return {
          ...baseParams,
          kind: 'onnx_to_trt_int8',
          output_path: params.output_path || 'best_int8.engine',
          calib_dir: params.calib_dir || '/app/int8_calib_images',
          precision: 'int8',
          device: params.device || 'gpu',
          mixed_fp16: params.mixed_fp16 || false,
          sparse: params.sparse || false,
          int8_max_batches: params.int8_max_batches || 10,
          input_size: params.input_size || [640, 640],
          workspace_mib: params.workspace_mib || 2048
        };
        
      case 'prune_unstructured':
        return {
          ...baseParams,
          kind: 'prune_unstructured',
          output_path: params.output_path || 'best_pruned_unstructured.pt',
          amount: params.amount || 0.2,
          pruning_type: params.pruning_type || 'l1_unstructured'
        };
        
      case 'prune_structured':
        return {
          ...baseParams,
          kind: 'prune_structured',
          output_path: params.output_path || 'best_pruned_structured.pt',
          amount: params.amount || 0.2,
          pruning_type: params.pruning_type || 'ln_structured',
          n: params.n || 2,
          dim: params.dim || 0
        };
        
      case 'check_model_stats':
        return {
          ...baseParams,
          kind: 'check_model_stats'
        };
        
      default:
        return baseParams;
    }
  };

  // 최적화 타입별 파라미터 추가
  baseRequest.parameters = getOptimizationParams(optimizationType, params);

  return baseRequest;
};

/**
 * 최적화 실행 함수
 */
export const runOptimization = async (optimizationType, params, pid, uid = '0001') => {
  console.log('runOptimization called with:', { optimizationType, params, pid, uid });
  
  const requestData = createOptimizationRequest(optimizationType, params, pid, uid);
  console.log('Created request data:', requestData);
  
  switch (optimizationType) {
    case 'pt_to_onnx_fp32':
      return await convertPtToOnnxFp32(requestData, uid);
    case 'pt_to_onnx_fp16':
      return await convertPtToOnnxFp16(requestData, uid);
    case 'onnx_to_trt':
      return await convertOnnxToTrt(requestData, uid);
    case 'onnx_to_trt_int8':
      return await convertOnnxToTrtInt8(requestData, uid);
    case 'prune_unstructured':
      return await pruneUnstructured(requestData, uid);
    case 'prune_structured':
      return await pruneStructured(requestData, uid);
    case 'check_model_stats':
      return await checkModelStats(requestData, uid);
    default:
      throw new Error(`Unknown optimization type: ${optimizationType}`);
  }
};
