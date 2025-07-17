import React from 'react';
import styles from './SnapshotSelector.module.css';
import { defaultSnapshot } from '../../hooks/useTrainingState.js';

const SnapshotSelector = ({ 
  snapshots, 
  selectedSnapshot, 
  onSnapshotChange,
  isRequired = false 
}) => {
  return (
    <div className={styles.selectorBox}>
      <label className={styles.paramLabel} style={{marginBottom: 4}}>
        {isRequired ? 'Snapshot' : 'Snapshot'} 
        {isRequired && <span style={{color:'#e74c3c'}}>*</span>}
      </label>
      <div className={styles.snapshotRow}>
        <select
          className={styles.select}
          value={selectedSnapshot ? selectedSnapshot.id : 'default'}
          onChange={e => {
            const snap = snapshots.find(s => s.id === e.target.value) || defaultSnapshot;
            onSnapshotChange(snap.id === 'default' ? null : snap);
          }}
        >
          {snapshots.map(snap => (
            <option key={snap.id} value={snap.id}>{snap.name}</option>
          ))}
        </select>
      </div>
      {selectedSnapshot && (
        <div className={styles.snapshotInfo}>
          <div><b>Name:</b> {selectedSnapshot.name}</div>
          <div><b>Description:</b> {selectedSnapshot.description}</div>
        </div>
      )}
    </div>
  );
};

export default SnapshotSelector; 