import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import styles from './ModelSelector.module.css';

const ModelSelector = ({ 
  modelType,
  onModelTypeChange,
  algorithm,
  onAlgorithmChange,
  customModel,
  onCustomModelChange,
  projectId = 'P0001',
  disabled = false,
  projectLoading = false
}) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModelData, setSelectedModelData] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      if (!projectId) {
        setModels([]);
        return;
      }
      
      setLoading(true);
      setError(null);
      try {
        console.log('Fetching custom models for projectId:', projectId);
        const result = await getTrainingList({ uid });
        console.log('Raw API response:', result);
        
        // projectId로 필터링하고 completed 상태의 모델만
        const filteredModels = result.filter(training => {
          const projectMatch = training.pid === projectId;
          const statusMatch = training.status === 'completed';
          
          console.log('Training item:', {
            tid: training.tid,
            pid: training.pid,
            projectId: projectId,
            status: training.status,
            projectMatch,
            statusMatch
          });
          
          return projectMatch && statusMatch;
        });
        
        console.log('Filtered models:', filteredModels);
        setModels(filteredModels);
      } catch (err) {
        setError(err.message);
        console.error('Failed to fetch custom models:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchModels();
  }, [projectId]);

  // 선택된 모델이 변경될 때 해당 모델의 데이터 저장
  useEffect(() => {
    if (customModel) {
      const modelData = models.find(model => model.tid === customModel);
      setSelectedModelData(modelData);
    } else {
      setSelectedModelData(null);
    }
  }, [customModel, models]);

  const handleModelTypeChange = (type) => {
    if (!disabled) {
      onModelTypeChange(type);
      // Model Type 변경 시 Custom Model 초기화
      if (type === 'pretrained') {
        onCustomModelChange('');
      }
    }
  };

  const handleCustomModelChange = (event) => {
    const selectedValue = event.target.value;
    onCustomModelChange(selectedValue);
  };

  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel}>Model</label>
      
      {/* Model Type Selection */}
      <div className={styles.typeOptions}>
        <button
          type="button"
          className={`${styles.typeOption} ${modelType === 'pretrained' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('pretrained')}
          disabled={disabled}
        >
          Pretrained Model
        </button>
        <button
          type="button"
          className={`${styles.typeOption} ${modelType === 'custom' ? styles.selected : ''}`}
          onClick={() => handleModelTypeChange('custom')}
          disabled={disabled}
        >
          Custom Model
        </button>
      </div>
      
      {/* Model Selector based on type */}
      {modelType === 'pretrained' ? (
        <div className={styles.pretrainedModelSection}>
          <label className={styles.subLabel}>Algorithm</label>
          <select
            className={styles.select}
            value={algorithm}
            onChange={e => onAlgorithmChange(e.target.value)}
            disabled={disabled}
          >
            <option value="">Select algorithm</option>
            <option value="yolov8n">YOLOv8n</option>
            <option value="yolov8s">YOLOv8s</option>
            <option value="yolov8m">YOLOv8m</option>
            <option value="yolov8l">YOLOv8l</option>
            <option value="yolov8x">YOLOv8x</option>
          </select>
        </div>
      ) : (
        <div className={styles.customModelSection}>
          <label className={styles.subLabel}>Custom Model</label>
          {projectLoading ? (
            <select disabled className={styles.select}>
              <option>Loading project information...</option>
            </select>
          ) : loading ? (
            <select disabled className={styles.select}>
              <option>Loading custom models...</option>
            </select>
          ) : error ? (
            <>
              <select disabled className={`${styles.select} ${styles.error}`}>
                <option>Error loading models</option>
              </select>
              <small className={styles.errorText}>{error}</small>
            </>
          ) : !projectId ? (
            <select disabled className={styles.select}>
              <option>Please set Project ID first</option>
            </select>
          ) : (
            <>
              <select
                value={customModel}
                onChange={handleCustomModelChange}
                className={styles.select}
                disabled={disabled}
              >
                <option value="">Select a custom model</option>
                {models.map(model => (
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
      )}
    </div>
  );
};

export default ModelSelector;
