import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const MetricSelector = ({ value, onChange, disabled, options }) => {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Metric</label>
      <select
        className={dsStyles.select}
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="">Select metric</option>
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
};

export default MetricSelector; 