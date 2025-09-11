import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { getOptimizationList } from '../../../api/optimization.js';
import { uid } from '../../../api/uid.js';
import styles from './OptimizationModelSelector.module.css';

// 최적화 타입별 지원 파일 형식 정보
const OPTIMIZATION_TYPE_CONSTRAINTS = {
  'pt_to_onnx_fp32': { supportedFormats: ['.pt'], description: 'PyTorch .pt 파일만 지원' },
  'pt_to_onnx_fp16': { supportedFormats: ['.pt'], description: 'PyTorch .pt 파일만 지원' },
  'onnx_to_trt': { supportedFormats: ['.onnx'], description: 'ONNX .onnx 파일만 지원' },
  'onnx_to_trt_int8': { supportedFormats: ['.onnx'], description: 'ONNX .onnx 파일만 지원' },
  'prune_unstructured': { supportedFormats: ['.pt'], description: 'PyTorch .pt 파일만 지원' },
  'prune_structured': { supportedFormats: ['.pt'], description: 'PyTorch .pt 파일만 지원' },
  'check_model_stats': { supportedFormats: ['.pt', '.onnx', '.engine'], description: 'PT/ONNX/TRT 모든 형식 지원' }
};

const OptimizationModelSelector = ({ 
  selectedModelType, 
  selectedModelId, 
  onModelTypeChange, 
  onModelIdChange,
  optimizationType,
  disabled = false,
  setRefreshCallback,
  projectId = 'P0001'
}) => {
  const [trainingModels, setTrainingModels] = useState([]);
  const [optimizationModels, setOptimizationModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModelData, setSelectedModelData] = useState(null);

  // 현재 제약 조건 정보 (useEffect보다 먼저 정의)
  const currentConstraints = optimizationType ? OPTIMIZATION_TYPE_CONSTRAINTS[optimizationType] : null;

  // 최적화 타입에 따른 모델 필터링 함수
  const getFilteredModels = (models, modelType) => {
    if (!optimizationType || !OPTIMIZATION_TYPE_CONSTRAINTS[optimizationType]) {
      return models;
    }

    const constraints = OPTIMIZATION_TYPE_CONSTRAINTS[optimizationType];
    
    return models.filter(model => {
      // Training 모델은 항상 .pt 파일 (best.pt)
      if (modelType === 'training') {
        return constraints.supportedFormats.includes('.pt');
      }
      
      // Optimization 모델은 kind에 따라 필터링
      if (modelType === 'optimization') {
        const modelKind = model.kind;
        if (!modelKind) return false;
        
        // kind를 파일 확장자로 매핑
        const kindToExtension = {
          'pt_to_onnx': '.onnx',
          'onnx_to_trt': '.engine',
          'onnx_to_trt_int8': '.engine',
          'prune_unstructured': '.pt',
          'prune_structured': '.pt',
          'check_model_stats': null // 모든 형식 지원
        };
        
        const modelExtension = kindToExtension[modelKind];
        if (modelExtension === null) return true; // check_model_stats는 모든 형식 지원
        return constraints.supportedFormats.includes(modelExtension);
      }
      
      return true;
    });
  };

  // 최적화 타입이 변경될 때 선택된 모델 ID가 호환되지 않으면 자동으로 리셋
  useEffect(() => {
    if (optimizationType && selectedModelId) {
      const isTrainingCompatible = currentConstraints && currentConstraints.supportedFormats.includes('.pt');
      const isOptimizationCompatible = currentConstraints && (
        currentConstraints.supportedFormats.includes('.onnx') || 
        currentConstraints.supportedFormats.includes('.engine') ||
        currentConstraints.supportedFormats.includes('.pt') // PT 모델도 optimization 모델에서 지원
      );
      
      // 현재 선택된 모델 타입이 호환되지 않으면 모델 ID 리셋
      if ((selectedModelType === 'training' && !isTrainingCompatible) ||
          (selectedModelType === 'optimization' && !isOptimizationCompatible)) {
        onModelIdChange('');
      }
    }
  }, [optimizationType, selectedModelType, selectedModelId, currentConstraints, onModelIdChange]);

  // 모델 리스트 가져오기 함수
  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      // 병렬로 두 API 호출하여 로딩 시간 단축
      const [trainingResult, optimizationResult] = await Promise.all([
        getTrainingList({ uid }),
        getOptimizationList({ uid })
      ]);

      const completedTrainingModels = trainingResult.filter(training => 
        training.status === 'completed' && training.pid === projectId
      );
      setTrainingModels(completedTrainingModels);

      const completedOptimizationModels = optimizationResult.filter(opt => 
        opt.status === 'completed' && opt.pid === projectId
      );
      setOptimizationModels(completedOptimizationModels);
      
      console.log('Models loaded successfully:', {
        trainingCount: completedTrainingModels.length,
        optimizationCount: completedOptimizationModels.length
      });
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch models:', err);
    } finally {
      setLoading(false);
    }
  };

  // Training list와 Optimization list 가져오기
  useEffect(() => {
    fetchModels();
  }, [projectId]);

  // Optimization type 변경 시 모델 리스트 새로고침
  useEffect(() => {
    if (optimizationType) {
      console.log('Optimization type changed, refreshing model list:', optimizationType);
      fetchModels();
    }
  }, [optimizationType, projectId]);

  // 새로고침 콜백 설정
  useEffect(() => {
    if (setRefreshCallback) {
      setRefreshCallback(fetchModels);
    }
  }, [setRefreshCallback]);

  // 선택된 모델이 변경될 때 해당 모델의 데이터 저장
  useEffect(() => {
    if (selectedModelId) {
      let modelData = null;
      if (selectedModelType === 'training') {
        modelData = trainingModels.find(model => model.tid === selectedModelId);
      } else if (selectedModelType === 'optimization') {
        modelData = optimizationModels.find(model => model.oid === selectedModelId);
      }
      setSelectedModelData(modelData);
    } else {
      setSelectedModelData(null);
    }
  }, [selectedModelId, selectedModelType, trainingModels, optimizationModels]);

  const handleModelTypeChange = (type) => {
    if (!disabled) {
      onModelTypeChange(type);
      // Model Type 변경 시 Model ID 초기화
      onModelIdChange('');
    }
  };

  const handleModelIdChange = (event) => {
    const selectedValue = event.target.value;
    console.log('Model ID changed:', {
      selectedValue,
      selectedValueType: typeof selectedValue,
      selectedValueLength: selectedValue ? selectedValue.length : 0
    });
    onModelIdChange(selectedValue);
  };

  // 필터링된 모델들
  const filteredTrainingModels = getFilteredModels(trainingModels, 'training');
  const filteredOptimizationModels = getFilteredModels(optimizationModels, 'optimization');

  return (
    <div className={styles.selectorBox}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <label className={styles.paramLabel}>Model</label>
        <button
          type="button"
          onClick={fetchModels}
          disabled={loading}
          className={styles.refreshBtn}
        >
          {loading ? 'REFRESHING...' : 'REFRESH'}
        </button>
      </div>
      
      
      {/* Model Type Selection */}
      <div className={styles.typeOptions}>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedModelType === 'training' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('training')}
          disabled={disabled || (currentConstraints && !currentConstraints.supportedFormats.includes('.pt'))}
        >
          Training Model
        </button>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedModelType === 'optimization' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('optimization')}
          disabled={disabled || (currentConstraints && !currentConstraints.supportedFormats.includes('.onnx') && !currentConstraints.supportedFormats.includes('.engine') && !currentConstraints.supportedFormats.includes('.pt'))}
        >
          Optimization Model
        </button>
      </div>
      
      {/* Model Selector based on type */}
      {selectedModelType === 'training' ? (
        <div className={styles.trainingModelSection}>
          <label className={styles.subLabel}>Training Model</label>
          {loading ? (
            <select disabled className={styles.select}>
              <option>Loading training models...</option>
            </select>
          ) : error ? (
            <>
              <select disabled className={`${styles.select} ${styles.error}`}>
                <option>Error loading models</option>
              </select>
              <small className={styles.errorText}>{error}</small>
            </>
          ) : (
            <>
              <select
                value={selectedModelId}
                onChange={handleModelIdChange}
                className={styles.select}
                disabled={disabled || loading}
              >
                <option value="">
                  {loading ? 'Loading training models...' : 
                   filteredTrainingModels.length === 0 ? 'No compatible training models available' :
                   'Select a training model'}
                </option>
                {filteredTrainingModels.map(model => (
                  <option key={model.tid} value={model.tid}>
                    {model.tid} - {model.origin_tid || model.parameters?.model || 'Unknown'} 
                    ({model.origin_dataset_name || 'Unknown Dataset'}) [.pt]
                  </option>
                ))}
              </select>
              
              {/* 선택된 모델의 classes 정보 표시 */}
              {selectedModelData && selectedModelData.classes && (
                <div className={styles.classesInfo}>
                  <div><b>Classes:</b> {selectedModelData.classes.join(', ')}</div>
                </div>
              )}
            </>
          )}
        </div>
      ) : (
        <div className={styles.optimizationModelSection}>
          <label className={styles.subLabel}>Optimization Model</label>
          {loading ? (
            <select disabled className={styles.select}>
              <option>Loading optimization models...</option>
            </select>
          ) : error ? (
            <>
              <select disabled className={`${styles.select} ${styles.error}`}>
                <option>Error loading models</option>
              </select>
              <small className={styles.errorText}>{error}</small>
            </>
          ) : (
            <>
              <select
                value={selectedModelId}
                onChange={handleModelIdChange}
                className={styles.select}
                disabled={disabled || loading}
              >
                <option value="">
                  {loading ? 'Loading optimization models...' : 
                   filteredOptimizationModels.length === 0 ? 'No compatible optimization models available' :
                   'Select an optimization model'}
                </option>
                {filteredOptimizationModels.map(model => {
                  const getFileFormat = (kind) => {
                    switch(kind) {
                      case 'pt_to_onnx': return '[.onnx]';
                      case 'onnx_to_trt':
                      case 'onnx_to_trt_int8': return '[.engine]';
                      case 'prune_unstructured':
                      case 'prune_structured': return '[.pt]';
                      case 'check_model_stats': return '[stats]';
                      default: return '';
                    }
                  };
                  
                  return (
                    <option key={model.oid} value={model.oid}>
                      {model.oid} - {model.kind?.replace(/_/g, ' ').toUpperCase() || 'Unknown'} {getFileFormat(model.kind)}
                    </option>
                  );
                })}
              </select>
              
              {/* 선택된 모델의 정보 표시 */}
              {selectedModelData && selectedModelData.metrics && (
                <div className={styles.classesInfo}>
                  <div><b>Type:</b> {selectedModelData.kind?.replace(/_/g, ' ').toUpperCase()}</div>
                  {selectedModelData.metrics.precision && (
                    <div><b>Precision:</b> {selectedModelData.metrics.precision.toUpperCase()}</div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default OptimizationModelSelector;
