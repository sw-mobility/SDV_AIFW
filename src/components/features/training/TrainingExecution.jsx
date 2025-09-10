import React from 'react';
import Button from '../../ui/atoms/Button.jsx';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import StatusBadge from '../validation/StatusBadge.jsx';
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
  onRunTraining,
  status,
  completeText,
  trainingResponse,
  modelType,
  customModel
}) => {
  return (
    <div className={styles.executionSection}>
      <div className={styles.executionHeader}>
        <div className={styles.executionTitle}>
          <h2 style={{ fontSize: 22, marginBottom: 0 }}>Training Execution</h2>
        </div>
        <StatusBadge status={status} />
      </div>
      
      <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="primary"
          size="medium"
          onClick={onRunTraining}
          disabled={isTraining || status === 'running'}
          className={styles.runButton}
        >
          {(isTraining || status === 'running') ? 'Running...' : 'Run Training'}
        </Button>
      </div>
      
      {/* Model Type Info */}
      {(modelType || customModel) && (
        <div className={styles.modelInfo}>
          <span className={styles.label}>Model Type:</span>
          <span className={styles.value}>{modelType === 'pretrained' ? 'Pretrained Model' : 'Custom Model'}</span>
          {modelType === 'custom' && customModel && (
            <span className={styles.value}> ({customModel})</span>
          )}
        </div>
      )}
      {status !== 'idle' && (
        <div className={styles.progressSection}>
          <ProgressBar 
            percentage={progress} 
            status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
            runningText="Training in progress..."
            completeText={completeText || "Training completed!"}
            errorText="Training failed."
          />
        </div>
      )}
        
      {/* Training Response Info */}
      {trainingResponse && (
        <div className={styles.responseInfo}>
          <h4>Training Details</h4>
          <div className={styles.responseDetails}>
            {trainingResponse.message && (
              <div className={styles.responseItem}>
                <span className={styles.label}>Status:</span>
                <span className={styles.value}>{trainingResponse.message}</span>
              </div>
            )}
            
            {trainingResponse.data?.parameters && (
              <>
                <div className={styles.responseItem}>
                  <span className={styles.label}>Model:</span>
                  <span className={styles.value}>{trainingResponse.data.parameters.model}</span>
                </div>
                <div className={styles.responseItem}>
                  <span className={styles.label}>Epochs:</span>
                  <span className={styles.value}>{trainingResponse.data.parameters.epochs}</span>
                </div>
                <div className={styles.responseItem}>
                  <span className={styles.label}>Batch Size:</span>
                  <span className={styles.value}>{trainingResponse.data.parameters.batch}</span>
                </div>
                <div className={styles.responseItem}>
                  <span className={styles.label}>Device:</span>
                  <span className={styles.value}>{trainingResponse.data.parameters.device}</span>
                </div>
              </>
            )}
            
            {trainingResponse.data?.user_classes && trainingResponse.data.user_classes.length > 0 && (
              <div className={styles.responseItem}>
                <span className={styles.label}>Classes:</span>
                <span className={styles.value}>{trainingResponse.data.user_classes.join(', ')}</span>
              </div>
            )}
            
            {trainingResponse.data?.cid && (
              <div className={styles.responseItem}>
                <span className={styles.label}>Codebase ID:</span>
                <span className={styles.value}>{trainingResponse.data.cid}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingExecution; 