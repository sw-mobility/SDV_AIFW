import React, { createContext, useContext, useState, useEffect } from 'react';
import { fetchRawDatasets, fetchLabeledDatasets } from '../api/datasets';
import { uid } from '../api/uid';

const DatasetContext = createContext();

export const useDatasetContext = () => useContext(DatasetContext);

export const DatasetProvider = ({ children }) => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const [rawResult, labeledResult] = await Promise.all([
        fetchRawDatasets({ uid }),
        fetchLabeledDatasets({ uid })
      ]);
      const raw = (rawResult.data || []).map(ds => ({ ...ds, datasetType: 'raw' }));
      const labeled = (labeledResult.data || []).map(ds => ({ ...ds, datasetType: 'labeled' }));
      setDatasets([...raw, ...labeled]);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  return (
    <DatasetContext.Provider value={{ datasets, setDatasets, loading, error, reload: loadDatasets }}>
      {children}
    </DatasetContext.Provider>
  );
}; 