import React, { createContext, useContext, useState, useEffect } from 'react';
import { fetchAllDatasets } from '../api/datasets';

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
      const result = await fetchAllDatasets();
      const raw = (result.data.raw || []).map(ds => ({ ...ds, datasetType: 'raw' }));
      const labeled = (result.data.labeled || []).map(ds => ({ ...ds, datasetType: 'labeled' }));
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