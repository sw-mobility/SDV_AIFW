import React from 'react';
import styles from './Validation.module.css';

const StatusBadge = ({ status }) => {
  const statusConfig = {
    idle: { label: 'Ready', className: styles.statusIdle },
    running: { label: 'Running', className: styles.statusRunning },
    success: { label: 'Completed', className: styles.statusSuccess },
    error: { label: 'Failed', className: styles.statusError }
  };
  
  const config = statusConfig[status] || statusConfig.idle;
  
  return (
    <div className={styles.statusBadge}>
      <span className={config.className}>{config.label}</span>
    </div>
  );
};

export default StatusBadge; 