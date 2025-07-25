import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const boards = [
  { value: 'board1', label: 'Board 1' },
  { value: 'board2', label: 'Board 2' },
  { value: 'board3', label: 'Board 3' },
];

export default function TargetBoardSelector({ value, onChange, disabled }) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Target Board</label>
      <select
        className={dsStyles.select}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="">Select board</option>
        {boards.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
} 