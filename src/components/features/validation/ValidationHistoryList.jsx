import React, { useState, useEffect } from 'react';
import { getValidationList } from '../../../api/validation.js';
import { uid } from '../../../api/uid.js';
import styles from './ValidationHistoryList.module.css';

const ValidationHistoryList = ({ onRefresh }) => {
  const [validations, setValidations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedValidation, setSelectedValidation] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const fetchValidations = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getValidationList({ uid });
      console.log('Raw validation list:', result);
      
      if (result && Array.isArray(result)) {
        // 최근 생성된 validation이 가장 위에 오도록 정렬
        const sortedValidations = result.sort((a, b) => {
          const dateA = new Date(a.created_at || 0);
          const dateB = new Date(b.created_at || 0);
          return dateB - dateA; // 내림차순 (최신순)
        });
        
        console.log('Sorted validations:', sortedValidations);
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
  };

  useEffect(() => {
    fetchValidations();
  }, []);

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
    if (metrics.total_classes !== undefined) {
      metricItems.push(`Classes: ${metrics.total_classes}`);
    }
    
    return metricItems.length > 0 ? metricItems.join(', ') : '-';
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
            onClick={onRefresh || fetchValidations} 
            className={styles.refreshBtn}
          >
            REFRESH
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
                    {validation.pid && <span className={styles.projectId}>P: {validation.pid}</span>}
                  </td>
                  <td className={styles.modelCell}>
                    <span className={styles.modelName}>{getModelDisplay(validation)}</span>
                    {validation.used_codebase && (
                      <span className={styles.codebaseTag}>{validation.used_codebase}</span>
                    )}
                  </td>
                  <td className={styles.datasetCell}>
                    <span className={styles.datasetName}>{getDatasetDisplay(validation)}</span>
                    {validation.classes && validation.classes.length > 0 && (
                      <span className={styles.classesCount}>
                        {validation.classes.length} classes
                      </span>
                    )}
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
        />
      )}
    </div>
  );
};

// Validation Detail Modal Component
const ValidationDetailModal = ({ validation, isOpen, onClose }) => {
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
    
    // Class information
    if (metrics.total_classes !== undefined || metrics.class_names) {
      const classInfo = [];
      if (metrics.total_classes !== undefined) classInfo.push(`Total Classes: ${metrics.total_classes}`);
      if (metrics.class_names && Object.keys(metrics.class_names).length > 0) {
        const classNames = Object.values(metrics.class_names).join(', ');
        classInfo.push(`Class Names: ${classNames}`);
      }
      
      if (classInfo.length > 0) {
        sections.push({
          title: 'Class Information',
          items: classInfo
        });
      }
    }
    
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
                  <label>Classes ({validation.classes.length}):</label>
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
