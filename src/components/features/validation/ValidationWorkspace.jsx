import React from 'react';
import Button from '../../ui/Button.jsx';
import ProgressBar from '../../ui/ProgressBar.jsx';
import StatusBadge from './StatusBadge.jsx';
import ResultsTable from './ResultsTable.jsx';
import styles from './Validation.module.css';

const ValidationWorkspace = ({ 
  status, 
  progress, 
  onRunValidation, 
  isDisabled, 
  isRunning,
  results
}) => {
  return (
    <div className={styles.workspace}>
      {/* 상단: 실행 섹션 */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.pageTitle} style={{ fontSize: 22, marginBottom: 0 }}>Validation Execution</h2>
        </div>
        <StatusBadge status={status} />
      </div>
      
      <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="primary"
          size="medium"
          onClick={onRunValidation}
          disabled={isDisabled}
          className={styles.runButton}
        >
          {isRunning ? 'Running...' : 'Run Validation'}
        </Button>
      </div>
      
      {status !== 'idle' && (
        <div className={styles.progressSection}>
          <ProgressBar
            percentage={progress}
            status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
            completeText="Validation completed!"
          />
        </div>
      )}
      
      {/* 하단: 결과 섹션 */}
      {results.length > 0 && (
        <div className={styles.resultsSection}>
          <h3 style={{ marginBottom: 16 }}>Results</h3>
          <ResultsTable results={results} />
        </div>
      )}
    </div>
  );
};

export default ValidationWorkspace; 