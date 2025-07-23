import React from 'react';
import ProgressBar from '../common/ProgressBar.jsx';
import Button from '../common/Button.jsx';
import styles from '../training/TrainingExecution.module.css';

const statusText = {
  idle: 'Ready.',
  running: 'optimization in progress...',
  success: 'Optimization completed!',
  error: 'Optimization failed.'
};

const OptimizationExecution = ({ isRunning, progress, logs, status, onRun }) => (
  <div className={styles.executionCard} style={{ maxWidth: 420, margin: '0 auto', marginTop: 24 }}>
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18 }}>
      <Button
        size="large"
        variant="primary"
        onClick={onRun}
        disabled={isRunning}
        style={{ minWidth: 180, fontWeight: 600, fontSize: 16 }}
      >
        {isRunning ? 'Running...' : 'Run Optimization'}
      </Button>
      <ProgressBar
        label="Progress"
        percentage={progress || 0}
        status={status}
        completeText={statusText.success}
      />
    </div>
    <div className={styles.logBox} style={{ marginTop: 24, minHeight: 60 }}>
      {(logs && logs.length > 0) ? (
        logs.map((log, i) => (
          <div key={i} style={{ fontSize: 13, color: '#444', marginBottom: 2 }}>{log}</div>
        ))
      ) : (
        <span style={{ color: '#bbb', fontSize: 13 }}>No logs yet.</span>
      )}
    </div>
  </div>
);

export default OptimizationExecution; 