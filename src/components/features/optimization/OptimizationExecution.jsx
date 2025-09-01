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
  results,
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
      
      {/* Results Table - Validation과 완전히 동일한 스타일 */}
      {status === 'success' && results && results.length > 0 && (
        <div style={{ marginTop: '32px' }}>
          <h3 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
            Optimization Results
          </h3>
          <div style={{ 
            overflowX: 'auto', 
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #e2e8f0',
            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
          }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}>
              <thead>
                <tr style={{ 
                  background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
                  borderBottom: '2px solid #e2e8f0'
                }}>
                  <th style={{ 
                    padding: '16px 12px', 
                    textAlign: 'left',
                    fontWeight: '600',
                    fontSize: '14px',
                    color: '#374151',
                    borderBottom: '2px solid #e2e8f0'
                  }}>
                    Optimization ID
                  </th>
                  <th style={{ 
                    padding: '16px 12px', 
                    textAlign: 'left',
                    fontWeight: '600',
                    fontSize: '14px',
                    color: '#374151',
                    borderBottom: '2px solid #e2e8f0'
                  }}>
                    Type
                  </th>
                  <th style={{ 
                    padding: '16px 12px', 
                    textAlign: 'left',
                    fontWeight: '600',
                    fontSize: '14px',
                    color: '#374151',
                    borderBottom: '2px solid #e2e8f0'
                  }}>
                    Parameters
                  </th>
                  <th style={{ 
                    padding: '16px 12px', 
                    textAlign: 'left',
                    fontWeight: '600',
                    fontSize: '14px',
                    color: '#374151',
                    borderBottom: '2px solid #e2e8f0'
                  }}>
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {results
                  .sort((a, b) => {
                    // Optimization ID를 숫자로 변환하여 내림차순 정렬
                    const aId = parseInt(a.oid?.replace('O', '') || '0');
                    const bId = parseInt(b.oid?.replace('O', '') || '0');
                    return bId - aId; // 내림차순 (큰 숫자가 위로)
                  })
                  .map((result, index) => (
                  <tr key={result.oid || index} style={{ 
                    borderBottom: '1px solid #f1f5f9',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f8fafc';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                  >
                    <td style={{ 
                      padding: '16px 12px',
                      fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
                      fontSize: '13px',
                      fontWeight: '500',
                      color: '#1f2937'
                    }}>
                      {result.oid || 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '16px 12px',
                      fontSize: '14px',
                      color: '#374151',
                      fontWeight: '500',
                      textTransform: 'capitalize'
                    }}>
                      {result.kind ? result.kind.replace(/_/g, ' ') : 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '16px 12px',
                      fontSize: '13px',
                      color: '#6b7280',
                      maxWidth: '300px',
                      wordBreak: 'break-word'
                    }}>
                      {formatParameters(optimizationParams)}
                    </td>
                    <td style={{ padding: '16px 12px' }}>
                      <span style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '600',
                        textTransform: 'capitalize',
                        background: result.service_response?.status === 'started' ? '#dbeafe' : 
                                   result.service_response?.status === 'failed' ? '#fef2f2' : '#dcfce7',
                        color: result.service_response?.status === 'started' ? '#1d4ed8' : 
                               result.service_response?.status === 'failed' ? '#dc2626' : '#16a34a',
                        border: result.service_response?.status === 'started' ? '1px solid #bfdbfe' : 
                                result.service_response?.status === 'failed' ? '1px solid #fecaca' : '1px solid #bbf7d0'
                      }}>
                        {result.service_response?.status || 'N/A'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptimizationExecution;
