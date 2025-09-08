import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import StatusBadge from '../validation/StatusBadge.jsx';
import styles from './OptimizationExecution.module.css';

/**
 * Optimization 실행 및 결과 표시 컴포넌트
 * Validation의 ResultsTable과 완전히 동일한 스타일
 */
const OptimizationExecution = ({
  isRunning,
  progress,
  status,
  onRunOptimization,
  optimizationType,
  optimizationParams
}) => {
  const canRun = !isRunning && optimizationType;

  // 파라미터 정보를 문자열로 변환하는 함수
  const formatParameters = (params) => {
    if (!params || Object.keys(params).length === 0) return 'N/A';
    
    const paramItems = [];
    
    // 주요 파라미터들만 표출
    if (params.input_size) {
      paramItems.push(`Input Size: [${params.input_size.join(', ')}]`);
    }
    if (params.batch_size) {
      paramItems.push(`Batch Size: ${params.batch_size}`);
    }
    if (params.channels) {
      paramItems.push(`Channels: ${params.channels}`);
    }
    if (params.precision) {
      paramItems.push(`Precision: ${params.precision}`);
    }
    if (params.device) {
      paramItems.push(`Device: ${params.device}`);
    }
    if (params.amount) {
      paramItems.push(`Amount: ${params.amount}`);
    }
    if (params.pruning_type) {
      paramItems.push(`Pruning Type: ${params.pruning_type.replace(/_/g, ' ')}`);
    }
    if (params.n) {
      paramItems.push(`L-norm: ${params.n}`);
    }
    if (params.dim) {
      paramItems.push(`Dimension: ${params.dim}`);
    }
    if (params.calib_dir) {
      paramItems.push(`Calib Dir: ${params.calib_dir.split('/').pop()}`);
    }
    if (params.workspace_mib) {
      paramItems.push(`Workspace: ${params.workspace_mib} MiB`);
    }
    
    return paramItems.length > 0 ? paramItems.join(', ') : 'Default parameters';
  };

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
          {status === 'running' ? 'Processing...' : 'Run Optimization'}
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
      

    </div>
  );
};

export default OptimizationExecution;
