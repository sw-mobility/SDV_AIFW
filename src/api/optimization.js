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
 * PT to ONNX FP32 변환
 */
export const convertPtToOnnxFp32 = async (data, uid = '0001') => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimizing/pt_to_onnx_fp32`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    if (response.status === 202) {
      return await response.json();
    } else if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.error('PT to ONNX FP32 conversion failed:', error);
    throw error;
  }
};

/**
 * PT to ONNX FP16 변환
 */
export const convertPtToOnnxFp16 = async (data, uid = '0001') => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimizing/pt_to_onnx_fp16`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    if (response.status === 202) {
      return await response.json();
    } else if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.error('PT to ONNX FP16 conversion failed:', error);
    throw error;
  }
};

/**
 * 비정형 프루닝
 */
export const pruneUnstructured = async (data, uid = '0001') => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimizing/prune_unstructured`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    if (response.status === 202) {
      return await response.json();
    } else if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Unstructured pruning failed:', error);
    throw error;
  }
};

/**
 * 정형 프루닝
 */
export const pruneStructured = async (data, uid = '0001') => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimizing/prune_structured`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    if (response.status === 202) {
      return await response.json();
    } else if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Structured pruning failed:', error);
    throw error;
  }
};

/**
 * 모델 통계 확인
 */
export const checkModelStats = async (data, uid = '0001') => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimizing/check_model_stats`, {
      method: 'POST',
      headers: getHeaders(uid),
      body: JSON.stringify(data)
    });

    if (response.status === 202) {
      return await response.json();
    } else if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Model stats check failed:', error);
    throw error;
  }
};

/**
 * 최적화 요청 데이터 생성 헬퍼 함수
 */
export const createOptimizationRequest = (optimizationType, params, pid, oid, uid = '0001') => {
  const workdir = `artifacts/${pid}/optimizing/${oid}`;
  
  // 경로 자동 생성
  const input_path = `artifacts/${pid}/training/${params.training_id || 'T0001'}/${params.model_name || 'yolov8n.pt'}`;
  
  // output_path는 최적화 타입에 따라 자동 생성
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
  
  const baseRequest = {
    pid,
    oid,
    action: optimizationType,
    parameters: {
      info: {
        uid,
        pid,
        oid,
        action: optimizationType,
        workdir
      },
      kind: optimizationType,
      input_path,
      output_path: getOutputPath()
    },
    info: {
      uid,
      pid,
      oid,
      action: optimizationType,
      workdir
    }
  };

  // 최적화 타입별 파라미터 추가
  switch (optimizationType) {
    case 'pt_to_onnx_fp32':
    case 'pt_to_onnx_fp16':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        input_size: params.input_size || [640, 640],
        batch_size: params.batch_size || 1,
        opset_version: params.opset_version || 11,
        dynamic_axes: params.dynamic_axes || false
      };
      break;
      
    case 'prune_unstructured':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        amount: params.amount || 0.2,
        pruning_type: params.pruning_type || 'l1_unstructured',
        global_unstructured: params.global_unstructured || false,
        importance: params.importance || 'magnitude'
      };
      break;
      
    case 'prune_structured':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        amount: params.amount || 0.2,
        pruning_type: params.pruning_type || 'ln_structured',
        n: params.n || 2,
        dim: params.dim || 0
      };
      break;
      
    case 'check_model_stats':
      baseRequest.parameters = {
        ...baseRequest.parameters,
        save_stats: params.save_stats !== undefined ? params.save_stats : true,
        detailed_stats: params.detailed_stats || false,
        save_format: params.save_format || 'json'
      };
      break;
  }

  return baseRequest;
};

/**
 * 최적화 실행 함수
 */
export const runOptimization = async (optimizationType, params, pid, oid, uid = '0001') => {
  const requestData = createOptimizationRequest(optimizationType, params, pid, oid, uid);
  
  switch (optimizationType) {
    case 'pt_to_onnx_fp32':
      return await convertPtToOnnxFp32(requestData, uid);
    case 'pt_to_onnx_fp16':
      return await convertPtToOnnxFp16(requestData, uid);
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
