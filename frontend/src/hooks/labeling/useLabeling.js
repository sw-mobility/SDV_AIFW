import { useState, useEffect, useCallback } from 'react';
import { fetchRawDatasets, uid } from '../../api/index.js';

export const useLabeling = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [labelingFormat, setLabelingFormat] = useState('yolo'); // 기본값을 yolo로 설정
  const [modelType, setModelType] = useState('detection'); // 기본값을 detection으로 설정

  // 데이터셋 목록 조회
  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRawDatasets({ uid });
      const camelDatasets = (res.data || []).map(ds => ({
        ...ds,
        id: ds._id,
        createdAt: ds.created_at ? new Date(ds.created_at).toISOString().slice(0, 10) : undefined
      }));
      setDatasets(camelDatasets);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // 초기 로드
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  return {
    datasets,
    loading,
    error,
    selectedDataset,
    datasetLoading: loading,
    datasetError: error,
    labelingFormat,
    modelType,

    setSelectedDataset,
    setLabelingFormat,
    setModelType,
    fetchDatasets
  };
}; 