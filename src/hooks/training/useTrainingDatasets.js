import { useState, useEffect, useCallback } from 'react';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

export const useTrainingDatasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  const fetchDatasets = useCallback(async () => {
    setDatasetLoading(true);
    setDatasetError(null);
    try {
      const res = await fetchLabeledDatasets({ uid });
      const formattedDatasets = (res.data || []).map(ds => ({
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
      setDatasets(formattedDatasets);
      return formattedDatasets;
    } catch (err) {
      setDatasetError(err.message);
      throw err;
    } finally {
      setDatasetLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
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