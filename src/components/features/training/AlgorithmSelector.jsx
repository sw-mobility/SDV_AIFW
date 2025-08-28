import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import styles from './AlgorithmSelector.module.css';
import { algorithmOptions } from '../../../domain/training/parameterGroups.js';

/**
 * : 알고리즘 선택 컴포넌트
 * 주요 기능: 훈련에 사용할 알고리즘 선택
 * @param algorithm
 * @param onAlgorithmChange
 * @returns {Element}
 * @constructor
 */
const AlgorithmSelector = ({
  algorithm,
  onAlgorithmChange
}) => {
  return (
        <div className={styles.selectorBox}>
          <label className={styles.paramLabel} style={{marginBottom: 4}}>Model</label>
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