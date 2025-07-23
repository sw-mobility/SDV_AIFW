import React from 'react';
import styles from '../training/AlgorithmSelector.module.css';

const testDatasets = [
  { value: 'test1', label: 'Test Dataset 1' },
  { value: 'test2', label: 'Test Dataset 2' },
  { value: 'test3', label: 'Test Dataset 3' },
];

const TestDatasetSelector = ({ value, onChange }) => (
  <div className={styles.selectorBox} style={{ marginBottom: 20 }}>
    <label className={styles.paramLabel} style={{ marginBottom: 4 }}>Test Dataset</label>
    <select
      className={styles.select}
      value={value || ''}
      onChange={e => onChange(e.target.value)}
    >
      <option value='' disabled>Select Test Dataset</option>
      {testDatasets.map(d => (
        <option key={d.value} value={d.value}>{d.label}</option>
      ))}
    </select>
  </div>
);

export default TestDatasetSelector; 