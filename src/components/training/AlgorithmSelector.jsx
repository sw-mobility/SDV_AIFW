import React from 'react';
import Button from '../common/Button.jsx';
import styles from './AlgorithmSelector.module.css';
import { algorithmOptions } from '../../services/trainingService.js';

const AlgorithmSelector = ({ 
  algorithm, 
  onAlgorithmChange, 
  onShowCodeEditor 
}) => {
  return (
    <div className={styles.sectionCard}>
      <div className={styles.selectorGroup}>
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
      </div>
      {/* Edit Code/Expert Mode button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px' }}>
        <Button
            variant="secondary"
            onClick={onShowCodeEditor}
            style={{ minWidth: 140 }}
        >
          Edit Code (Expert Mode)
        </Button>
      </div>
    </div>
  );
};

export default AlgorithmSelector; 