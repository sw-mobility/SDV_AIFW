import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import styles from './CustomModelSelector.module.css';

const CustomModelSelector = ({ 
  selectedModel, 
  onModelChange, 
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
    if (selectedModel) {
      const modelData = models.find(model => model.tid === selectedModel);
      setSelectedModelData(modelData);
    } else {
      setSelectedModelData(null);
    }
  }, [selectedModel, models]);

  const handleChange = (event) => {
    const selectedValue = event.target.value;
    onModelChange(selectedValue);
  };

  if (projectLoading) {
    return (
      <div className={styles.selectorBox}>
        <label className={styles.paramLabel} style={{marginBottom: 4}}>Custom Model</label>
        <select disabled className={styles.select}>
          <option>Loading project information...</option>
        </select>
      </div>
    );
  }

  if (loading) {
    return (
      <div className={styles.selectorBox}>
        <label className={styles.paramLabel} style={{marginBottom: 4}}>Custom Model</label>
        <select disabled className={styles.select}>
          <option>Loading custom models...</option>
        </select>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.selectorBox}>
        <label className={styles.paramLabel} style={{marginBottom: 4}}>Custom Model</label>
        <select disabled className={`${styles.select} ${styles.error}`}>
          <option>Error loading models</option>
        </select>
        <small className={styles.errorText}>{error}</small>
      </div>
    );
  }

  if (!projectId) {
    return (
      <div className={styles.selectorBox}>
        <label className={styles.paramLabel} style={{marginBottom: 4}}>Custom Model</label>
        <select disabled className={styles.select}>
          <option>Please set Project ID first</option>
        </select>
      </div>
    );
  }

  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel} style={{marginBottom: 4}}>Custom Model</label>
      <select
        value={selectedModel}
        onChange={handleChange}
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
    </div>
  );
};

export default CustomModelSelector;
