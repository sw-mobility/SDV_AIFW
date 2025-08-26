import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const defaultModels = [
  { value: 'modelA', label: 'Model A' },
  { value: 'modelB', label: 'Model B' },
  { value: 'modelC', label: 'Model C' },
];

export default function ModelSelector({ 
  value, 
  onChange, 
  disabled, 
  models = defaultModels,
  showInfo = true 
}) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Model</label>
      <select
        className={dsStyles.select}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="">Select model</option>
        {models.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      {showInfo && (
        <div className={dsStyles.infoText}>
          Select a trained model for validation
        </div>
      )}
    </div>
  );
} 