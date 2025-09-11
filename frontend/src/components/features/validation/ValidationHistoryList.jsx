import React, { useState, useEffect, useCallback } from 'react';
import { getValidationList } from '../../../api/validation.js';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import styles from './ValidationHistoryList.module.css';

const ValidationHistoryList = ({ onRefresh, projectId }) => {
  const [validations, setValidations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedValidation, setSelectedValidation] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [trainings, setTrainings] = useState([]);
  const [trainingLoading, setTrainingLoading] = useState(false);

  const fetchValidations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getValidationList({ uid });
      console.log('Raw validation list:', result);
      
      if (result && Array.isArray(result)) {
        // 현재 프로젝트로 필터링
        const filteredValidations = projectId 
          ? result.filter(validation => validation.pid === projectId)
          : result;
        
        // 최근 생성된 validation이 가장 위에 오도록 정렬
        const sortedValidations = filteredValidations.sort((a, b) => {
          const dateA = new Date(a.created_at || 0);
          const dateB = new Date(b.created_at || 0);
          return dateB - dateA; // 내림차순 (최신순)
        });
        
        console.log('Filtered and sorted validations:', sortedValidations);
        setValidations(sortedValidations);
      } else {
        console.warn('Validation list is not an array:', result);
        setValidations([]);
      }
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch validation list:', err);
    } finally {
      setLoading(false);
    }
  }, [uid, projectId]);

  const fetchTrainings = useCallback(async () => {
    setTrainingLoading(true);
    try {
      const result = await getTrainingList({ uid });
      console.log('Raw training list:', result);
      
      if (result && Array.isArray(result)) {
        setTrainings(result);
      } else {
        console.warn('Training list is not an array:', result);
        setTrainings([]);
      }
    } catch (err) {
      console.error('Error fetching trainings:', err);
      setTrainings([]);
    } finally {
      setTrainingLoading(false);
    }
  }, [uid]);

  // onRefresh prop이 있을 때는 그것을 사용, 없을 때는 자체 fetchValidations 사용
  const handleRefresh = useCallback(async () => {
    if (onRefresh) {
      console.log('Using onRefresh prop for refresh');
      await onRefresh();
      // onRefresh 후에 ValidationHistoryList도 새로고침
      await fetchValidations();
    } else {
      console.log('Using local fetchValidations for refresh');
      await fetchValidations();
    }
  }, [onRefresh, fetchValidations]);

  useEffect(() => {
    fetchValidations();
  }, [uid, fetchValidations, onRefresh]); // uid, fetchValidations, onRefresh가 변경될 때마다 refresh

  useEffect(() => {
    fetchTrainings();
  }, [uid, fetchTrainings]); // training list도 가져오기

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      const date = new Date(dateString);
      return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      'completed': { color: 'success', label: 'Completed' },
      'running': { color: 'warning', label: 'Running' },
      'failed': { color: 'error', label: 'Failed' },
      'pending': { color: 'info', label: 'Pending' },
      'stopped': { color: 'default', label: 'Stopped' }
    };
    
    const config = statusConfig[status] || { color: 'default', label: status };
    
    return (
      <span className={`${styles.statusBadge} ${styles[config.color]}`}>
        {config.label}
      </span>
    );
  };

  const getModelDisplay = (validation) => {
    if (validation.parameters?.model) {
      return validation.parameters.model;
    }
    return validation.used_codebase || '-';
  };

  const getDatasetDisplay = (validation) => {
    if (validation.dataset_name) {
      return validation.dataset_name;
    }
    if (validation.did) {
      return validation.did;
    }
    return '-';
  };

  const getMetricsDisplay = (validation) => {
    if (!validation.metrics_summary) return '-';
    
    const metrics = validation.metrics_summary;
    const metricItems = [];
    
    if (metrics['mAP_0.5'] !== undefined) {
      metricItems.push(`mAP@0.5: ${metrics['mAP_0.5'].toFixed(3)}`);
    }
    if (metrics['mAP_0.5_0.95'] !== undefined) {
      metricItems.push(`mAP@0.5:0.95: ${metrics['mAP_0.5_0.95'].toFixed(3)}`);
    }
    
    return metricItems.length > 0 ? metricItems.join(', ') : '-';
  };

  // validation에서 사용한 training의 tid를 통해 해당 training의 classes를 찾는 함수
  const getModelClasses = (validation) => {
    console.log('=== getModelClasses Debug ===');
    console.log('validation:', validation);
    console.log('trainings.length:', trainings.length);
    console.log('trainings:', trainings);
    
    if (!trainings.length) {
      console.log('No trainings available');
      return null;
    }
    
    // validation에서 사용한 training ID 찾기
    // 1. validation.tid (validation 실행 시 사용한 training ID)
    // 2. validation.parameters.tid (validation parameters에 저장된 training ID)
    // 3. validation.parameters.model (model이 training ID인 경우)
    let trainingId = null;
    
    console.log('validation.tid:', validation.tid);
    console.log('validation.parameters?.tid:', validation.parameters?.tid);
    console.log('validation.parameters?.model:', validation.parameters?.model);
    
    if (validation.tid) {
      trainingId = validation.tid;
      console.log('Using validation.tid:', trainingId);
    } else if (validation.parameters?.tid) {
      trainingId = validation.parameters.tid;
      console.log('Using validation.parameters.tid:', trainingId);
    } else if (validation.parameters?.model && validation.parameters.model.startsWith('T')) {
      trainingId = validation.parameters.model;
      console.log('Using validation.parameters.model:', trainingId);
    }
    
    console.log('Final trainingId:', trainingId);
    
    if (trainingId) {
      const training = trainings.find(t => t.tid === trainingId);
      console.log('Found training:', training);
      if (training && training.classes && training.classes.length > 0) {
        console.log('Returning classes:', training.classes);
        return training.classes;
      } else {
        console.log('Training found but no classes:', training?.classes);
      }
    } else {
      console.log('No trainingId found');
    }
    
    console.log('Returning null');
    return null;
  };

  const handleRowClick = (validation) => {
    setSelectedValidation(validation);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedValidation(null);
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading validation history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.errorMessage}>Error: {error}</p>
        <button onClick={fetchValidations} className={styles.retryBtn}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.title}>Validation History</h3>
          <span className={styles.count}>({validations.length})</span>
        </div>
        <div className={styles.headerRight}>
          <button 
            onClick={handleRefresh} 
            className={styles.refreshBtn}
            disabled={loading}
          >
            {loading ? 'REFRESHING...' : 'REFRESH'}
          </button>
        </div>
      </div>
      
      {validations.length === 0 ? (
        <div className={styles.emptyState}>
          <h4>No validation history found</h4>
          <p>Run a validation to see results here.</p>
        </div>
      ) : (
        <div className={styles.tableContainer}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Validation ID</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Metrics</th>
                <th>Status</th>
                <th>Created At</th>
              </tr>
            </thead>
            <tbody>
              {validations.map((validation) => (
                <tr 
                  key={validation.vid} 
                  className={styles.tableRow}
                  onClick={() => handleRowClick(validation)}
                >
                  <td className={styles.vidCell}>
                    <span className={styles.vidValue}>{validation.vid}</span>
                  </td>
                  <td className={styles.modelCell}>
                    <span className={styles.modelName}>{getModelDisplay(validation)}</span>
                    {validation.used_codebase && (
                      <span className={styles.codebaseTag}>{validation.used_codebase}</span>
                    )}
                  </td>
                  <td className={styles.datasetCell}>
                    <span className={styles.datasetName}>{getDatasetDisplay(validation)}</span>
                  </td>
                  <td className={styles.metricsCell}>
                    {getMetricsDisplay(validation)}
                  </td>
                  <td>{getStatusBadge(validation.status)}</td>
                  <td className={styles.dateCell}>{formatDate(validation.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Validation Detail Modal */}
      {isModalOpen && selectedValidation && (
        <ValidationDetailModal
          validation={selectedValidation}
          isOpen={isModalOpen}
          onClose={closeModal}
          getModelClasses={getModelClasses}
        />
      )}
    </div>
  );
};

// Validation Detail Modal Component
const ValidationDetailModal = ({ validation, isOpen, onClose, getModelClasses }) => {
  if (!isOpen) return null;

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      const date = new Date(dateString);
      return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  const formatMetrics = (metrics) => {
    if (!metrics) return null;
    
    const sections = [];
    
    // Basic metrics
    if (metrics['mAP_0.5'] !== undefined || metrics['mAP_0.5_0.95'] !== undefined) {
      const basicMetrics = [];
      if (metrics['mAP_0.5'] !== undefined) basicMetrics.push(`mAP@0.5: ${metrics['mAP_0.5'].toFixed(4)}`);
      if (metrics['mAP_0.5_0.95'] !== undefined) basicMetrics.push(`mAP@0.5:0.95: ${metrics['mAP_0.5_0.95'].toFixed(4)}`);
      if (metrics.mean_precision !== undefined) basicMetrics.push(`Precision: ${metrics.mean_precision.toFixed(4)}`);
      if (metrics.mean_recall !== undefined) basicMetrics.push(`Recall: ${metrics.mean_recall.toFixed(4)}`);
      
      if (basicMetrics.length > 0) {
        sections.push({
          title: 'Basic Metrics',
          items: basicMetrics
        });
      }
    }
    
    // Inference speed
    if (metrics.inference_speed) {
      const speedMetrics = [];
      if (metrics.inference_speed.preprocess !== undefined) speedMetrics.push(`Preprocess: ${metrics.inference_speed.preprocess.toFixed(2)}ms`);
      if (metrics.inference_speed.inference !== undefined) speedMetrics.push(`Inference: ${metrics.inference_speed.inference.toFixed(2)}ms`);
      if (metrics.inference_speed.postprocess !== undefined) speedMetrics.push(`Postprocess: ${metrics.inference_speed.postprocess.toFixed(2)}ms`);
      
      if (speedMetrics.length > 0) {
        sections.push({
          title: 'Inference Speed',
          items: speedMetrics
        });
      }
    }
    
    // Class information은 Model Information과 Dataset Information 섹션에서 별도로 표시하므로 제거
    
    return sections;
  };

  const metricsSections = formatMetrics(validation.metrics_summary);

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Validation Details - {validation.vid}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>
        
        <div className={styles.modalBody}>
          <div className={styles.detailSection}>
            <h3>Basic Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Validation ID:</label>
                <span>{validation.vid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Project ID:</label>
                <span>{validation.pid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Status:</label>
                <span>{validation.status}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Created At:</label>
                <span>{formatDate(validation.created_at)}</span>
              </div>
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Model Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Model:</label>
                <span>{validation.parameters?.model || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Codebase:</label>
                <span>{validation.used_codebase || '-'}</span>
              </div>
              {(() => {
                console.log('=== Validation Detail Modal Debug ===');
                console.log('validation in modal:', validation);
                const modelClasses = getModelClasses(validation);
                console.log('modelClasses in modal:', modelClasses);
                return modelClasses && modelClasses.length > 0 && (
                  <div className={styles.detailItem}>
                    <label>Model Classes ({modelClasses.length}):</label>
                    <span>{modelClasses.join(', ')}</span>
                  </div>
                );
              })()}
              {validation.artifacts_path && (
                <div className={styles.detailItem}>
                  <label>Artifacts Path:</label>
                  <span>{validation.artifacts_path}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Dataset Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Dataset:</label>
                <span>{validation.dataset_name || validation.did || '-'}</span>
              </div>
              {validation.classes && validation.classes.length > 0 && (
                <div className={styles.detailItem}>
                  <label>Dataset Classes ({validation.classes.length}):</label>
                  <span>{validation.classes.join(', ')}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Validation Parameters</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Image Size:</label>
                <span>{validation.parameters?.imgsz || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Batch Size:</label>
                <span>{validation.parameters?.batch || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Device:</label>
                <span>{validation.parameters?.device || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Confidence:</label>
                <span>{validation.parameters?.conf || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>IoU:</label>
                <span>{validation.parameters?.iou || '-'}</span>
              </div>
            </div>
          </div>

          {metricsSections && metricsSections.length > 0 && (
            <div className={styles.detailSection}>
              <h3>Performance Metrics</h3>
              {metricsSections.map((section, index) => (
                <div key={index} className={styles.metricsSection}>
                  <h4>{section.title}</h4>
                  <div className={styles.metricsGrid}>
                    {section.items.map((item, itemIndex) => (
                      <div key={itemIndex} className={styles.metricItem}>
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {validation.error_details && (
            <div className={styles.detailSection}>
              <h3>Error Details</h3>
              <div className={styles.errorDetails}>
                <pre>{validation.error_details}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ValidationHistoryList;
