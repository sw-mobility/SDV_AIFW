import React from 'react';
import styles from '../training/AlgorithmSelector.module.css';

const models = [
  { value: 'modelA', label: 'Model A' },
  { value: 'modelB', label: 'Model B' },
  { value: 'modelC', label: 'Model C' },
];

const ModelSelector = ({ value, onChange }) => (
  <div className={styles.selectorBox} style={{ marginBottom: 20 }}>
    <label className={styles.paramLabel} style={{ marginBottom: 4 }}>Model</label>
    <select
      className={styles.select}
      value={value || ''}
      onChange={e => onChange(e.target.value)}
    >
      <option value='' disabled>Select Model</option>
      {models.map(m => (
        <option key={m.value} value={m.value}>{m.label}</option>
      ))}
    </select>
  </div>
);

export default ModelSelector; 