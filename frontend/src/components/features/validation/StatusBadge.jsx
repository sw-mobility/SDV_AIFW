import React from 'react';
import styles from './Validation.module.css';

/**
 * 기능: 검증 상태 배지
 * 주요 기능: 검증 진행 상태를 시각적으로 표시
 * @param status
 * @returns {Element}
 * @constructor
 */
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