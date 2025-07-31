import { useState, useCallback } from 'react';
import { createRawDataset, createLabeledDataset } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

const DATASET_TYPES = [
  'Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'
];

export const useDatasetUpload = (initialData = {}, editMode = false, datasetType = 'raw', onCreated) => {
  // 편집 모드일 때만 initialData 사용, 그렇지 않으면 빈 값으로 초기화
  const [formData, setFormData] = useState(() => {
    if (editMode && initialData && initialData.name) {
      return {
        name: initialData.name || '',
        type: initialData.type || DATASET_TYPES[0],
        description: initialData.description || '',
        taskType: initialData.task_type || initialData.taskType || 'Classification',
        labelFormat: initialData.label_format || initialData.labelFormat || 'COCO'
      };
    }
    return {
      name: '',
      type: DATASET_TYPES[0],
      description: '',
      taskType: 'Classification',
      labelFormat: 'COCO'
    };
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  // 폼 데이터 업데이트
  const updateFormData = useCallback((field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  // 제출 처리
  const handleSubmit = useCallback(async (e) => {
    if (e) e.preventDefault();
    
    setLoading(true);
    setError(null);
    setSuccess(false);
    
    try {
      if (editMode) {
        // 편집 모드 - onSave 콜백 사용
        setSuccess(true);
        return;
      }
      
      if (datasetType === 'labeled') {
        await createLabeledDataset({
          uid: uid,
          name: formData.name,
          description: formData.description,
          type: formData.type,
          task_type: formData.taskType,
          label_format: formData.labelFormat
        });
      } else {
        // 메타 정보만 생성
        await createRawDataset({
          uid: uid,
          name: formData.name,
          description: formData.description,
          type: formData.type
        });
      }
      
      setSuccess(true);
      onCreated && onCreated();
      
      setTimeout(() => {
        setSuccess(false);
      }, 1000);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [formData, editMode, datasetType, onCreated]);

  return {
    // 상태
    formData,
    loading,
    error,
    success,
    
    // 핸들러
    updateFormData,
    handleSubmit,
    
    // 상수
    DATASET_TYPES
  };
}; 