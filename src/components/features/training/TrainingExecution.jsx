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
  onRunTraining,
  status,
  completeText,
  trainingResponse
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
          <ProgressBar 
            percentage={progress} 
            status={status} 
            runningText="Training in progress..."
            completeText={completeText || "Training completed!"}
            errorText="Training failed."
          />
        </div>
        
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
              

            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingExecution; 