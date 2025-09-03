import React, { useState, useEffect } from 'react';
import { getOptimizationList } from '../../../api/optimization.js';
import { uid } from '../../../api/uid.js';
import styles from './OptimizationHistoryList.module.css';

const OptimizationHistoryList = ({ onRefresh }) => {
  const [optimizations, setOptimizations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedOptimization, setSelectedOptimization] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const fetchOptimizations = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getOptimizationList({ uid });
      console.log('Raw optimization list:', result);
      
      if (result && Array.isArray(result)) {
        // 최근 생성된 optimization이 가장 위에 오도록 정렬
        const sortedOptimizations = result.sort((a, b) => {
          const dateA = new Date(a.started_at || 0);
          const dateB = new Date(b.started_at || 0);
          return dateB - dateA; // 내림차순 (최신순)
        });
        
        console.log('Sorted optimizations:', sortedOptimizations);
        setOptimizations(sortedOptimizations);
      } else {
        console.warn('Optimization list is not an array:', result);
        setOptimizations([]);
      }
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch optimization list:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOptimizations();
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

  const getOptimizationTypeDisplay = (optimization) => {
    if (optimization.kind) {
      return optimization.kind.replace(/_/g, ' ').toUpperCase();
    }
    return '-';
  };

  const getInputModelDisplay = (optimization) => {
    if (optimization.input_model_id) {
      return optimization.input_model_id;
    }
    if (optimization.parameters?.input_path) {
      // input_path에서 model ID 추출
      const pathParts = optimization.parameters.input_path.split('/');
      const modelId = pathParts[pathParts.length - 2]; // 마지막에서 두 번째 부분
      if (modelId && (modelId.startsWith('T') || modelId.startsWith('O'))) {
        return modelId;
      }
    }
    return '-';
  };

  const getMetricsDisplay = (optimization) => {
    if (!optimization.metrics) return '-';
    
    const metrics = optimization.metrics;
    const metricItems = [];
    
    if (metrics.precision) {
      metricItems.push(`Precision: ${metrics.precision.toUpperCase()}`);
    }
    
    if (metrics.stats) {
      if (metrics.stats.size_mb !== undefined) {
        metricItems.push(`Size: ${metrics.stats.size_mb.toFixed(2)}MB`);
      }
      if (metrics.stats.total_params !== undefined) {
        metricItems.push(`Params: ${metrics.stats.total_params.toLocaleString()}`);
      }
    }
    
    return metricItems.length > 0 ? metricItems.join(', ') : '-';
  };

  const handleRowClick = (optimization) => {
    setSelectedOptimization(optimization);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedOptimization(null);
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading optimization history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.errorMessage}>Error: {error}</p>
        <button onClick={fetchOptimizations} className={styles.retryBtn}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.title}>Optimization History</h3>
          <span className={styles.count}>({optimizations.length})</span>
        </div>
        <div className={styles.headerRight}>
          <button 
            onClick={onRefresh || fetchOptimizations} 
            className={styles.refreshBtn}
          >
            REFRESH
          </button>
        </div>
      </div>
      
      {optimizations.length === 0 ? (
        <div className={styles.emptyState}>
          <h4>No optimization history found</h4>
          <p>Run an optimization to see results here.</p>
        </div>
      ) : (
        <div className={styles.tableContainer}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Optimization ID</th>
                <th>Type</th>
                <th>Input Model</th>
                <th>Metrics</th>
                <th>Status</th>
                <th>Started At</th>
              </tr>
            </thead>
            <tbody>
              {optimizations.map((optimization) => (
                <tr 
                  key={optimization.oid} 
                  className={styles.tableRow}
                  onClick={() => handleRowClick(optimization)}
                >
                  <td className={styles.oidCell}>
                    <span className={styles.oidValue}>{optimization.oid}</span>
                    {optimization.pid && <span className={styles.projectId}>P: {optimization.pid}</span>}
                  </td>
                  <td className={styles.typeCell}>
                    <span className={styles.typeName}>{getOptimizationTypeDisplay(optimization)}</span>
                  </td>
                  <td className={styles.modelCell}>
                    <span className={styles.modelName}>{getInputModelDisplay(optimization)}</span>
                  </td>
                  <td className={styles.metricsCell}>
                    {getMetricsDisplay(optimization)}
                  </td>
                  <td>{getStatusBadge(optimization.status)}</td>
                  <td className={styles.dateCell}>{formatDate(optimization.started_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Optimization Detail Modal */}
      {isModalOpen && selectedOptimization && (
        <OptimizationDetailModal
          optimization={selectedOptimization}
          isOpen={isModalOpen}
          onClose={closeModal}
        />
      )}
    </div>
  );
};

// Optimization Detail Modal Component
const OptimizationDetailModal = ({ optimization, isOpen, onClose }) => {
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
    if (metrics.precision) {
      sections.push({
        title: 'Model Information',
        items: [`Precision: ${metrics.precision.toUpperCase()}`]
      });
    }
    
    // Stats
    if (metrics.stats) {
      const statsItems = [];
      if (metrics.stats.format) statsItems.push(`Format: ${metrics.stats.format.toUpperCase()}`);
      if (metrics.stats.size_mb !== undefined) statsItems.push(`Size: ${metrics.stats.size_mb.toFixed(2)}MB`);
      if (metrics.stats.total_params !== undefined) statsItems.push(`Total Parameters: ${metrics.stats.total_params.toLocaleString()}`);
      if (metrics.stats.nonzero_params !== undefined) statsItems.push(`Non-zero Parameters: ${metrics.stats.nonzero_params.toLocaleString()}`);
      if (metrics.stats.sparsity_pct !== undefined) statsItems.push(`Sparsity: ${metrics.stats.sparsity_pct.toFixed(2)}%`);
      
      if (statsItems.length > 0) {
        sections.push({
          title: 'Model Statistics',
          items: statsItems
        });
      }
    }
    
    // Artifacts
    if (metrics.artifact_files && metrics.artifact_files.length > 0) {
      sections.push({
        title: 'Artifact Files',
        items: metrics.artifact_files
      });
    }
    
    return sections;
  };

  const metricsSections = formatMetrics(optimization.metrics);

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Optimization Details - {optimization.oid}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>
        
        <div className={styles.modalBody}>
          <div className={styles.detailSection}>
            <h3>Basic Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Optimization ID:</label>
                <span>{optimization.oid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Project ID:</label>
                <span>{optimization.pid}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Status:</label>
                <span>{optimization.status}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Started At:</label>
                <span>{formatDate(optimization.started_at)}</span>
              </div>
              {optimization.completed_at && (
                <div className={styles.detailItem}>
                  <label>Completed At:</label>
                  <span>{formatDate(optimization.completed_at)}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Optimization Information</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <label>Type:</label>
                <span>{optimization.kind?.replace(/_/g, ' ').toUpperCase() || '-'}</span>
              </div>
              <div className={styles.detailItem}>
                <label>Input Model ID:</label>
                <span>{optimization.input_model_id || '-'}</span>
              </div>
              {optimization.artifacts_path && (
                <div className={styles.detailItem}>
                  <label>Artifacts Path:</label>
                  <span>{optimization.artifacts_path}</span>
                </div>
              )}
            </div>
          </div>

          <div className={styles.detailSection}>
            <h3>Parameters</h3>
            <div className={styles.detailGrid}>
              {optimization.parameters?.input_size && (
                <div className={styles.detailItem}>
                  <label>Input Size:</label>
                  <span>{optimization.parameters.input_size.join(' x ')}</span>
                </div>
              )}
              {optimization.parameters?.batch_size && (
                <div className={styles.detailItem}>
                  <label>Batch Size:</label>
                  <span>{optimization.parameters.batch_size}</span>
                </div>
              )}
              {optimization.parameters?.input_path && (
                <div className={styles.detailItem}>
                  <label>Input Path:</label>
                  <span>{optimization.parameters.input_path}</span>
                </div>
              )}
              {optimization.parameters?.output_path && (
                <div className={styles.detailItem}>
                  <label>Output Path:</label>
                  <span>{optimization.parameters.output_path}</span>
                </div>
              )}
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

          {optimization.error_details && (
            <div className={styles.detailSection}>
              <h3>Error Details</h3>
              <div className={styles.errorDetails}>
                <pre>{optimization.error_details}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptimizationHistoryList;
