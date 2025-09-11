import React from 'react';
import styles from './ModelTypeSelector.module.css';

const ModelTypeSelector = ({ selectedType, onTypeChange, disabled = false }) => {
  const handleTypeChange = (type) => {
    if (!disabled) {
      onTypeChange(type);
    }
  };

  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel}>Model Type</label>
      <div className={styles.typeOptions}>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedType === 'pretrained' ? styles.selected : ''}`}
          onClick={() => handleTypeChange('pretrained')}
          disabled={disabled}
        >
          Pretrained Model
        </button>
        <button
          type="button"
          className={`${styles.typeOption} ${selectedType === 'custom' ? styles.selected : ''}`}
          onClick={() => handleTypeChange('custom')}
          disabled={disabled}
        >
          Custom Model
        </button>
      </div>
    </div>
  );
};

export default ModelTypeSelector;
