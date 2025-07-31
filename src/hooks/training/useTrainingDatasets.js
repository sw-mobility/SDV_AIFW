import { useState, useEffect } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';
import { useAsync } from '../common/useAsync.js';

export const useTrainingDatasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  
  const { execute: fetchDatasets, status: datasetLoading, error: datasetError } = useAsync(
    async () => {
      const res = await fetchLabeledDatasets({ uid });
      return (res.data || []).map(ds => ({
        id: ds.did || ds._id,
        name: ds.name,
        type: ds.type,
        size: ds.total,
        labelCount: ds.total,
        description: ds.description,
        task_type: ds.task_type,
        label_format: ds.label_format,
        origin_raw: ds.origin_raw,
        created_at: ds.created_at,
      }));
    }
  );

  useEffect(() => {
    fetchDatasets().then(setDatasets);
  }, [fetchDatasets]);

  const selectDataset = (dataset) => {
    setSelectedDataset(dataset);
  };

  const clearSelectedDataset = () => {
    setSelectedDataset(null);
  };

  return {
    datasets,
    selectedDataset,
    setSelectedDataset: selectDataset,
    clearSelectedDataset,
    datasetLoading,
    datasetError,
    refetchDatasets: fetchDatasets
  };
}; 