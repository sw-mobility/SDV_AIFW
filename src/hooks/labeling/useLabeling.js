import { useState, useEffect, useCallback } from 'react';
import { fetchRawDatasets, uid } from '../../api';

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

    setSelectedDataset,
    fetchDatasets
  };
}; 