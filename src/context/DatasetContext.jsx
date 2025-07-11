import React, { createContext, useContext, useState, useEffect } from 'react';
import { fetchDatasetList } from '../api/dataset';

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
      const data = await fetchDatasetList();
      setDatasets(data);
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