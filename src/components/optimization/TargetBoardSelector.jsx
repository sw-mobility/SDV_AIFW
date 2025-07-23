import React from 'react';
import styles from '../training/AlgorithmSelector.module.css';

const boards = [
  { value: 'board1', label: 'Board 1' },
  { value: 'board2', label: 'Board 2' },
  { value: 'board3', label: 'Board 3' },
];

const TargetBoardSelector = ({ value, onChange }) => (
  <div className={styles.selectorBox} style={{ marginBottom: 20 }}>
    <label className={styles.paramLabel} style={{ marginBottom: 4 }}>Target Board</label>
    <select
      className={styles.select}
      value={value || ''}
      onChange={e => onChange(e.target.value)}
    >
      <option value='' disabled>Select Board</option>
      {boards.map(b => (
        <option key={b.value} value={b.value}>{b.label}</option>
      ))}
    </select>
  </div>
);

export default TargetBoardSelector; 