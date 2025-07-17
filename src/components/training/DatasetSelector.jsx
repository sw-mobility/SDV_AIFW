import React from 'react';
import Loading from '../common/Loading.jsx';
import styles from './DatasetSelector.module.css';

const DatasetSelector = ({ 
  datasets, 
  selectedDataset, 
  onDatasetChange, 
  datasetLoading, 
  datasetError 
}) => {
  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel} style={{marginBottom: 4}}>Dataset</label>
      {datasetLoading && <Loading />}
      {datasetError && <span className={styles.inputErrorMsg}>{datasetError}</span>}
      {!datasetLoading && !datasetError && (
        <select
          className={styles.select}
          value={selectedDataset ? selectedDataset.id : ''}
          onChange={e => {
            const ds = datasets.find(d => d.id === Number(e.target.value));
            onDatasetChange(ds);
          }}
        >
          <option value="">Select dataset</option>
          {datasets.map(ds => (
            <option key={ds.id} value={ds.id}>
              {ds.name} ({ds.type}, {ds.size})
            </option>
          ))}
        </select>
      )}
      {selectedDataset && (
        <div className={styles.datasetInfo}>
          <div><b>Name:</b> {selectedDataset.name}</div>
          <div><b>Type:</b> {selectedDataset.type}</div>
          <div><b>Size:</b> {selectedDataset.size}</div>
          <div><b>Label Count:</b> {selectedDataset.labelCount}</div>
        </div>
      )}
    </div>
  );
};

export default DatasetSelector; 