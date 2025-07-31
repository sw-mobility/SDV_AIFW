import { useState, useEffect, useCallback } from 'react';
import { fetchRawDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

export const useLabeling = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);

  // 데이터셋 목록 조회
  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRawDatasets({ uid });
      const camelDatasets = (res.data || []).map(ds => ({
        ...ds,
        id: ds.id || ds._id, // id 필드 보장
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
    // 상태
    datasets,
    loading,
    error,
    selectedDataset,
    
    // 핸들러
    setSelectedDataset,
    
    // 유틸리티
    fetchDatasets
  };
}; 