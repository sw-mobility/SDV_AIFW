import React from 'react';
import { Settings, Play, CheckCircle, XCircle } from 'lucide-react';
import ProgressBar from '../../ui/atoms/ProgressBar.jsx';
import Button from '../../ui/atoms/Button.jsx';
import styles from './LabelingWorkspace.module.css';
import { useLabelingWorkspace } from '../../../hooks/labeling/useLabelingWorkspace.js';

export default function LabelingWorkspace({ dataset }) {
  const {
    modelType,
    taskType,
    status,
    progress,
    handleRunLabeling,
    handleModelTypeChange,
    handleTaskTypeChange,
    isDisabled
  } = useLabelingWorkspace(dataset);

  if (!dataset) {
    return (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <Settings size={48} color="#cbd5e1" />
          </div>
          <div className={styles.emptyContent}>
            <h3 className={styles.emptyTitle}>No Dataset Selected</h3>
            <p className={styles.emptyText}>
              Please select a dataset from the left panel to start labeling
            </p>
          </div>
        </div>
    );
  }

  return (
      <div className={styles.workspace}>
        {/* 헤더 섹션 */}
        <div className={styles.header}>
          <div className={styles.datasetInfo}>
            <h2 className={styles.datasetName}>{dataset.name}</h2>
            <div className={styles.datasetMeta}>
              <span className={styles.metaItem}>{dataset.type}</span>
              <span className={styles.metaDivider}>•</span>
              <span className={styles.metaItem}>{dataset.size}</span>
            </div>
          </div>
          <div className={styles.statusBadge}>
            {status === 'idle' && <span className={styles.statusIdle}>Ready</span>}
            {status === 'running' && <span className={styles.statusRunning}>Running</span>}
            {status === 'success' && (
                <span className={styles.statusSuccess}>
              <CheckCircle size={16} />
              Completed
            </span>
            )}
            {status === 'error' && (
                <span className={styles.statusError}>
              <XCircle size={16} />
              Error
            </span>
            )}
          </div>
        </div>

        {/* 설정 섹션 */}
        <div className={styles.configSection}>
          <h3 className={styles.sectionTitle}>Configuration</h3>
          <div className={styles.configRow}>
            <div className={styles.configGroup}>
              <label className={styles.configLabel}>Model Type</label>
              <select
                  value={modelType}
                  onChange={e => handleModelTypeChange(e.target.value)}
                  className={styles.configSelect}
                  disabled={status === 'running'}
              >
                <option value="YOLO">YOLO</option>
                <option value="Faster R-CNN">Faster R-CNN</option>
                <option value="SSD">SSD</option>
              </select>
            </div>

            <div className={styles.configGroup}>
              <label className={styles.configLabel}>Task Type</label>
              <select
                  value={taskType}
                  onChange={e => handleTaskTypeChange(e.target.value)}
                  className={styles.configSelect}
                  disabled={status === 'running'}
              >
                <option value="Object detection">Object detection</option>
                <option value="Image classification">Image classification</option>
                <option value="Semantic segmentation">Semantic segmentation</option>
              </select>
            </div>

            <div className={styles.configGroup}>
              <Button
                  variant="primary"
                  size="medium"
                  onClick={handleRunLabeling}
                  disabled={status === 'running'}
                  className={styles.runButton}
              >
                {status === 'running' ? (
                    <>
                      <div className={styles.buttonSpinner} />
                      Running...
                    </>
                ) : (
                    <>Run Labeling</>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* 진행률 표시 */}
        {status !== 'idle' && (
            <div className={styles.progressSection}>
              <ProgressBar
                  percentage={progress}
                  status={status}
                  completeText="Labeling completed successfully!"
              />
            </div>
        )}

        {/* 에디터 영역 */}
        <div className={styles.editorSection}>
          <h3 className={styles.sectionTitle}>Labeling Editor</h3>
          <div className={styles.editorArea}>
            <div className={styles.editorPlaceholder}>
              <Settings size={32} color="#cbd5e1" />
              <p>Labeling editor will appear here</p>
              <small>Image annotation, text labeling, or other tools</small>
            </div>
          </div>
        </div>
      </div>
  );
}