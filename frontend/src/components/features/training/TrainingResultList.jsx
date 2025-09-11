import React, { useState, useEffect, useCallback } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import styles from './TrainingResultList.module.css';

const TrainingResultList = ({ onRefresh, projectId }) => {
  const [trainings, setTrainings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTraining, setSelectedTraining] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const fetchTrainings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getTrainingList({ uid });
      console.log('Raw training list:', result);
      
      if (result && Array.isArray(result)) {
        // 현재 프로젝트로 필터링
        const filteredTrainings = projectId 
          ? result.filter(training => training.pid === projectId)
          : result;
        
        // 최근 생성된 training이 가장 위에 오도록 정렬
        const sortedTrainings = filteredTrainings.sort((a, b) => {
          const dateA = new Date(a.started_at || a.created_at || 0);
          const dateB = new Date(b.started_at || b.created_at || 0);
          return dateB - dateA; // 내림차순 (최신순)
        });
        
        console.log('Filtered and sorted trainings:', sortedTrainings);
        setTrainings(sortedTrainings);
      } else {
        console.warn('Training list is not an array:', result);
        setTrainings([]);
      }
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch training list:', err);
    } finally {
      setLoading(false);
    }
  }, [uid, projectId]);

  useEffect(() => {
    fetchTrainings();
  }, [uid, fetchTrainings, onRefresh]);

  // onRefresh prop이 있을 때는 그것을 사용, 없을 때는 자체 fetchTrainings 사용
  const handleRefresh = useCallback(async () => {
    if (onRefresh) {
      console.log('Using onRefresh prop for refresh');
      await onRefresh();
      // onRefresh 후에 TrainingResultList도 새로고침
      await fetchTrainings();
    } else {
      console.log('Using local fetchTrainings for refresh');
      await fetchTrainings();
    }
  }, [onRefresh, fetchTrainings]);

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

  const getAlgorithmDisplay = (training) => {
    // origin_tid가 있으면 custom model, 없으면 pretrained model
    if (training.origin_tid && training.origin_tid !== training.parameters?.model) {
      return `Custom (${training.origin_tid})`;
    }
    return training.parameters?.model || training.origin_tid || '-';
  };

  const getDatasetDisplay = (training) => {
    if (training.origin_dataset_name) {
      return training.origin_dataset_name;
    }
    if (training.origin_did) {
      return training.origin_did;
    }
    return '-';
  };

  const handleRowClick = (training) => {
    setSelectedTraining(training);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedTraining(null);
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading training results...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.errorMessage}>Error: {error}</p>
        <button onClick={fetchTrainings} className={styles.retryBtn}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.title}>Training Results</h3>
          <span className={styles.count}>({trainings.length})</span>
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
      
      {trainings.length === 0 ? (
        <div className={styles.emptyState}>
          <h4>No training results found</h4>
          <p>Start a new training to see results here.</p>
        </div>
      ) : (
        <div className={styles.tableContainer}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Training ID</th>
                <th>Algorithm</th>
                <th>Dataset</th>
                <th>Codebase</th>
                <th>Status</th>
                <th>Started At</th>
                <th>Epochs</th>
              </tr>
            </thead>
            <tbody>
              {trainings.map((training) => (
                <tr 
                  key={training.tid} 
                  className={styles.tableRow}
                  onClick={() => handleRowClick(training)}
                >
                  <td className={styles.tidCell}>
                    <span className={styles.tidValue}>{training.tid}</span>
                  </td>
                  <td className={styles.algorithmCell}>
                    <span className={styles.algorithmName}>{getAlgorithmDisplay(training)}</span>
                    {training.parameters?.pretrained !== false && (
                      <span className={styles.pretrainedTag}>Pretrained</span>
                    )}
                  </td>
                  <td className={styles.datasetCell}>
                    <span className={styles.datasetName}>{getDatasetDisplay(training)}</span>
                    {training.classes && training.classes.length > 0 && (
                      <span className={styles.classesCount}>
                        {training.classes.length} classes
                      </span>
                    )}
                  </td>
                  <td className={styles.codebaseCell}>
                    {training.cid ? (
                      <span className={styles.codebaseId}>{training.cid}</span>
                    ) : (
                      <span className={styles.noCodebase}>-</span>
                    )}
                  </td>
                  <td>{getStatusBadge(training.status)}</td>
                  <td className={styles.dateCell}>{formatDate(training.started_at)}</td>
                  <td className={styles.epochsCell}>
                    <span className={styles.epochsValue}>
                      {training.parameters?.epochs || '-'}
                    </span>
                    {training.parameters?.batch && (
                      <span className={styles.batchInfo}>Batch: {training.parameters.batch}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Training Detail Modal */}
      {isModalOpen && selectedTraining && (
        <TrainingDetailModal
          training={selectedTraining}
          isOpen={isModalOpen}
          onClose={closeModal}
        />
      )}
    </div>
  );
};

// Training Detail Modal Component
const TrainingDetailModal = ({ training, isOpen, onClose }) => {
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

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Training Details - {training.tid}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>
        
        <div className={styles.modalBody}>
          <div className={styles.detailSection}>
            <h3>Basic Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Training ID:</label>
                <span>{training.tid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Project ID:</label>
                <span>{training.pid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Status:</label>
                <span>{training.status}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Started At:</label>
                <span>{formatDate(training.started_at)}</span>
              </div>
              {training.completed_at && (
                <div className={styles.detailItem}>
                  <label>Completed At:</label>
                  <span>{formatDate(training.completed_at)}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Model Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Model:</label>
                <span>{training.parameters?.model || training.origin_tid || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Model Type:</label>
                <span>{training.origin_tid && training.origin_tid !== training.parameters?.model ? 'Custom Model' : 'Pretrained Model'}</span>
              </div>
              {training.artifacts_path && (
                <div className={styles.detailItem}>
                  <label>Artifacts Path:</label>
                  <span>{training.artifacts_path}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Dataset Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Original Dataset:</label>
                <span>{training.origin_dataset_name || training.origin_did || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Processed Dataset:</label>
                <span>{training.processed_dataset_name || training.processed_did || '-'}</span>
              </div>
              {training.classes && training.classes.length > 0 && (
                <div className={styles.detailItem}>
                  <label>Classes ({training.classes.length}):</label>
                  <span>{training.classes.join(', ')}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Training Parameters</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Epochs:</label>
                <span>{training.parameters?.epochs || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Batch Size:</label>
                <span>{training.parameters?.batch || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Image Size:</label>
                <span>{training.parameters?.imgsz || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Device:</label>
                <span>{training.parameters?.device || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Optimizer:</label>
                <span>{training.parameters?.optimizer || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Learning Rate:</label>
                <span>{training.parameters?.lr0 || '-'}</span>
              </div>
              {training.cid && (
                <div className={styles.detailItem}>
                  <label>Codebase ID:</label>
                  <span>{training.cid}</span>
                </div>
              )}
            </div>
          </div>

          {training.error_details && (
            <div className={styles.detailSection}>
              <h3>Error Details</h3>
              <div className={styles.errorDetails}>
                <pre>{training.error_details}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingResultList;
