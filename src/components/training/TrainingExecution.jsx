import React from 'react';
import Button from '../common/Button.jsx';
import ProgressBar from '../common/ProgressBar.jsx';
import styles from './TrainingExecution.module.css';

const TrainingExecution = ({ 
  isTraining, 
  progress, 
  logs, 
  onRunTraining 
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
          <ProgressBar percentage={progress} />
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