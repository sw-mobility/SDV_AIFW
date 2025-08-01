import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import styles from './TrainingExecution.module.css';

/**
 * 훈련 실행 및 모니터링
 * 주요 기능:
 * 훈련 시작/중지
 * 진행률 표시
 * 훈련 상태 모니터링
 * @param isTraining
 * @param progress
 * @param logs
 * @param onRunTraining
 * @param status
 * @param completeText
 * @returns {Element}
 * @constructor
 */
const TrainingExecution = ({ 
  isTraining, 
  progress, 
  logs, 
  onRunTraining,
  status,
  completeText
}) => {
  return (
    <div className={styles.sectionCard}>
      <div className={styles.runCard}>
        <div className={styles.runRow}>
          <Button
              variant="primary-gradient"
              size="medium"
              onClick={onRunTraining}
              disabled={isTraining}
              icon={isTraining ? <span className={styles.spinner}></span> : null}
              style={{ minWidth: 150, opacity: isTraining ? 0.6 : 1, cursor: isTraining ? 'not-allowed' : 'pointer' }}
          >
            {isTraining ? 'Running...' : 'Run Training'}
          </Button>
        </div>
      </div>
      <div className={styles.statusCard}>
        <div>
          <ProgressBar percentage={progress} status={status} completeText={completeText} />
        </div>
        <div className={styles.logBox}>
          {logs.length === 0 ? (
              <span className={styles.logEmpty}>No logs yet.</span>
          ) : (
              logs.map((log, i) => <div key={i}>{log}</div>)
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingExecution; 