import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import StatusBadge from '../validation/StatusBadge.jsx';
import styles from './OptimizationExecution.module.css';

/**
 * Optimization 실행 및 결과 표시 컴포넌트
 * Validation 페이지와 동일한 구조
 */
const OptimizationExecution = ({
  isRunning,
  progress,
  status,
  logs,
  results,
  onRunOptimization,
  selectedModel,
  optimizationType
}) => {
  const canRun = !isRunning && selectedModel && optimizationType;

  return (
    <div className={styles.executionSection}>
      <div className={styles.executionHeader}>
        <div className={styles.executionTitle}>
          <h2 style={{ fontSize: 22, marginBottom: 0 }}>Optimization Execution</h2>
        </div>
        <StatusBadge status={status} />
      </div>
      
      <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="primary"
          size="medium"
          onClick={onRunOptimization}
          disabled={!canRun}
          className={styles.runButton}
        >
          {isRunning ? 'Running...' : 'Run Optimization'}
        </Button>
      </div>
      
      {status !== 'idle' && (
        <div className={styles.progressSection}>
          <ProgressBar
            percentage={progress}
            status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
            runningText="Optimization in progress..."
            completeText="Optimization completed!"
            errorText="Optimization failed."
          />
        </div>
      )}
      
      {/* Results Section */}
      {status === 'success' && results && (
        <div className={styles.resultsSection}>
          <h3 style={{ marginBottom: 16 }}>Results</h3>
          <div className={styles.resultsInfo}>
            <div className={styles.resultItem}>
              <span className={styles.resultLabel}>Output Model:</span>
              <span className={styles.resultValue}>{results.outputPath || 'Generated automatically'}</span>
            </div>
            {results.statsPath && (
              <div className={styles.resultItem}>
                <span className={styles.resultLabel}>Statistics:</span>
                <span className={styles.resultValue}>{results.statsPath}</span>
              </div>
            )}
            <div className={styles.resultItem}>
              <span className={styles.resultLabel}>Processing Time:</span>
              <span className={styles.resultValue}>{results.processingTime || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Logs Section */}
      {logs && logs.length > 0 && (
        <div className={styles.logsSection}>
          <h3 style={{ marginBottom: 16 }}>Logs</h3>
          <div className={styles.logsContainer}>
            {logs.map((log, index) => (
              <div key={index} className={styles.logLine}>
                {log}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default OptimizationExecution;
