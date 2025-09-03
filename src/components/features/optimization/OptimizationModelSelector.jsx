import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { getOptimizationList } from '../../../api/optimization.js';
import { uid } from '../../../api/uid.js';
import styles from './OptimizationModelSelector.module.css';

const OptimizationModelSelector = ({ 
  selectedModelType, 
  selectedModelId, 
  onModelTypeChange, 
  onModelIdChange,
  disabled = false 
}) => {
  const [trainingModels, setTrainingModels] = useState([]);
  const [optimizationModels, setOptimizationModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModelData, setSelectedModelData] = useState(null);

  // Training list와 Optimization list 가져오기
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      try {
        // Training list 가져오기
        const trainingResult = await getTrainingList({ uid });
        const completedTrainingModels = trainingResult.filter(training => 
          training.status === 'completed'
        );
        setTrainingModels(completedTrainingModels);

        // Optimization list 가져오기
        const optimizationResult = await getOptimizationList({ uid });
        const completedOptimizationModels = optimizationResult.filter(opt => 
          opt.status === 'completed'
        );
        setOptimizationModels(completedOptimizationModels);
      } catch (err) {
        setError(err.message);
        console.error('Failed to fetch models:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

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

  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel}>Model</label>
      
      {/* Model Type Selection */}
      <div className={styles.typeOptions}>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedModelType === 'training' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('training')}
          disabled={disabled}
        >
          Training Model
        </button>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedModelType === 'optimization' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('optimization')}
          disabled={disabled}
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
                disabled={disabled}
              >
                <option value="">Select a training model</option>
                {trainingModels.map(model => (
                  <option key={model.tid} value={model.tid}>
                    {model.tid} - {model.origin_tid || model.parameters?.model || 'Unknown'} 
                    ({model.origin_dataset_name || 'Unknown Dataset'})
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
                disabled={disabled}
              >
                <option value="">Select an optimization model</option>
                {optimizationModels.map(model => (
                  <option key={model.oid} value={model.oid}>
                    {model.oid} - {model.kind?.replace(/_/g, ' ').toUpperCase() || 'Unknown'}
                  </option>
                ))}
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
