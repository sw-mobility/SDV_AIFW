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
 * API 명세서에 따라 정리
 */
export const createOptimizationRequest = (optimizationType, params, pid, uid = '0001') => {
  // 기본 요청 구조 (API 명세서에 따라)
  const baseRequest = {
    pid,
    parameters: {
      kind: optimizationType,
      input_path: `artifacts/${pid}/training/${params.training_id || 'T0001'}/${params.model_name || 'best.pt'}`
    }
  };

  // 최적화 타입별 파라미터 추가
  switch (optimizationType) {
    case 'pt_to_onnx_fp32':
    case 'pt_to_onnx_fp16':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'pt_to_onnx', // API 명세서에 따라 'pt_to_onnx'로 통일
        output_path: params.output_path || `${params.model_name?.replace('.pt', '') || 'model'}_${optimizationType.includes('fp16') ? 'fp16' : 'fp32'}.onnx`,
        input_size: params.input_size || [640, 640],
        batch_size: params.batch_size || 1,
        channels: params.channels || 3
      };
      break;
      
    case 'onnx_to_trt':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'onnx_to_trt', // API 명세서에 따라 'onnx_to_trt'로 통일
        output_path: params.output_path || `${params.model_name?.replace('.onnx', '') || 'model'}_${params.precision || 'fp32'}.engine`,
        precision: params.precision || 'fp32', // 'fp32' or 'fp16'
        device: params.device || 'gpu' // 'gpu', 'dla', 'dla0', 'dla1'
      };
      break;
      
    case 'onnx_to_trt_int8':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'onnx_to_trt_int8', // API 명세서에 따라 'onnx_to_trt_int8'로 통일
        output_path: params.output_path || `${params.model_name?.replace('.onnx', '') || 'model'}_int8.engine`,
        calib_dir: params.calib_dir || '/app/int8_calib_images',
        precision: 'int8', // 고정값
        device: params.device || 'gpu', // 'gpu' or 'dla'
        mixed_fp16: params.mixed_fp16 || false,
        sparse: params.sparse || false,
        int8_max_batches: params.int8_max_batches || 10,
        input_size: params.input_size || [640, 640],
        workspace_mib: params.workspace_mib || 2048
      };
      break;
      
    case 'prune_unstructured':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'prune_unstructured', // API 명세서에 따라 'prune_unstructured'로 통일
        output_path: params.output_path || `${params.model_name?.replace('.pt', '') || 'model'}_pruned_unstructured.pt`,
        amount: params.amount || 0.2,
        pruning_type: params.pruning_type || 'l1_unstructured'
      };
      break;
      
    case 'prune_structured':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'prune_structured', // API 명세서에 따라 'prune_structured'로 통일
        output_path: params.output_path || `${params.model_name?.replace('.pt', '') || 'model'}_pruned_structured.pt`,
        amount: params.amount || 0.2,
        pruning_type: params.pruning_type || 'ln_structured',
        n: params.n || 2,
        dim: params.dim || 0
      };
      break;
      
    case 'check_model_stats':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        kind: 'check_model_stats' // API 명세서에 따라 'check_model_stats'로 통일
      };
      break;
  }

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
