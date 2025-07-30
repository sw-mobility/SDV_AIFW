import React from 'react';
import Button from '../../shared/common/Button.jsx';
import styles from './AlgorithmSelector.module.css';
import { algorithmOptions } from '../../domain/training/parameterGroups.js';

const AlgorithmSelector = ({ 
  algorithm, 
  onAlgorithmChange
}) => {
  return (
        <div className={styles.selectorBox}>
          <label className={styles.paramLabel} style={{marginBottom: 4}}>Algorithm</label>
          <select
            className={styles.select}
            value={algorithm}
            onChange={e => onAlgorithmChange(e.target.value)}
          >
            {algorithmOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
    </div>
  );
};

export default AlgorithmSelector; 