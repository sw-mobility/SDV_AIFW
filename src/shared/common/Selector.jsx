import React from 'react';
import styles from './Selector.module.css';

export default function Selector({ label, options, value, onChange, disabled, placeholder }) {
  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel}>{label}</label>
      <select
        className={styles.select}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
      >
        <option value="" disabled>{placeholder || `Select ${label}`}</option>
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
} 