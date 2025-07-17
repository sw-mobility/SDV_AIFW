import React from 'react';
import styles from './ParameterChipList.module.css';

const ParameterChipList = ({ paramGroups, algoParams, selectedKey, onSelect }) => {
  if (!paramGroups) return null;
  
  return (
    <div className={styles.paramChipListWrap}>
      {paramGroups.flatMap((group) =>
        group.params.map((param) => {
          const value = algoParams[param.key] ?? param.default;
          const isChanged = value !== param.default;
          const isSelected = selectedKey === param.key;
          return (
            <button
              key={param.key}
              className={
                styles.paramChip +
                (isChanged ? ' ' + styles.paramChipChanged : ' ' + styles.paramChipDefault) +
                (isSelected ? ' ' + styles.paramChipSelected : '')
              }
              onClick={() => onSelect(param.key)}
              type="button"
              tabIndex={0}
            >
              <span className={styles.paramChipLabel}>{param.label}</span>
              <span className={styles.paramChipValue}>{String(value)}</span>
            </button>
          );
        })
      )}
    </div>
  );
};

export default ParameterChipList; 