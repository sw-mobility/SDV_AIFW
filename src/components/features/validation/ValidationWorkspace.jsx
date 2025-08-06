import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import StatusBadge from './StatusBadge.jsx';
import ResultsTable from './ResultsTable.jsx';
import styles from './Validation.module.css';

/**
 * validation page 우측 필드
 * 기능: 검증 상태 배지
 * 주요 기능: 검증 진행 상태를 시각적으로 표시
 * @param status
 * @param progress
 * @param onRunValidation
 * @param isDisabled
 * @param isRunning
 * @param results
 * @returns {Element}
 * @constructor
 */
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
            runningText="Validation in progress..."
            completeText="Validation completed!"
            errorText="Validation failed."
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