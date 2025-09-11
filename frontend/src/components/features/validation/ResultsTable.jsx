import React, { useState } from 'react';
import MetricsDetailModal from './MetricsDetailModal.jsx';
import styles from './ResultsTable.module.css';

/**
 * 기능: 검증 결과 테이블
 * 주요 기능: 검증 결과를 테이블 형태로 표시
 * @param results
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ResultsTable = ({ results }) => {
  const [selectedMetrics, setSelectedMetrics] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  if (results.length === 0) return null;

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  const renderMetrics = (metrics) => {
    if (!metrics || typeof metrics !== 'object') return 'N/A';
    
    const metricItems = [];
    
    // 주요 metrics 표시 - API 응답 키에 맞게 수정
    if (metrics['mAP_0.5'] !== undefined) {
      const map05 = metrics['mAP_0.5'];
      metricItems.push(`mAP@0.5: ${map05.toFixed(4)}`);
    }
    if (metrics['mAP_0.5_0.95'] !== undefined) {
      const map0595 = metrics['mAP_0.5_0.95'];
      metricItems.push(`mAP@0.5:0.95: ${map0595.toFixed(4)}`);
    }
    if (metrics.mean_precision !== undefined) {
      metricItems.push(`Precision: ${metrics.mean_precision.toFixed(4)}`);
    }
    if (metrics.mean_recall !== undefined) {
      metricItems.push(`Recall: ${metrics.mean_recall.toFixed(4)}`);
    }
    if (metrics.total_classes !== undefined) {
      metricItems.push(`Classes: ${metrics.total_classes}`);
    }
    
    return metricItems.length > 0 ? metricItems.join(', ') : 'N/A';
  };

  const handleMetricsClick = (metrics) => {
    if (metrics && Object.keys(metrics).length > 0) {
      setSelectedMetrics(metrics);
      setIsModalOpen(true);
    }
  };

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Validation ID</th>
            <th>Model</th>
            <th>Dataset</th>
            <th>Metrics</th>
            <th>Status</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className={styles.tableRow}>
              <td className={styles.vidCell}>
                {r.vid || 'N/A'}
              </td>
              <td className={styles.modelCell}>
                {r.model}
              </td>
              <td className={styles.datasetCell}>
                {r.dataset}
              </td>
              <td 
                className={`${styles.metricsCell} ${r.metrics && Object.keys(r.metrics).length > 0 ? styles.clickable : ''}`}
                onClick={() => handleMetricsClick(r.metrics)}
                title={r.metrics && Object.keys(r.metrics).length > 0 ? "Click to view detailed metrics" : ""}
              >
                {renderMetrics(r.metrics)}
              </td>
              <td className={styles.statusCell}>
                <span className={`${styles.statusBadge} ${styles[r.status] || styles.default}`}>
                  {r.status || 'N/A'}
                </span>
              </td>
              <td className={styles.timestampCell}>
                {formatTimestamp(r.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Metrics Detail Modal */}
      <MetricsDetailModal
        metrics={selectedMetrics}
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setSelectedMetrics(null);
        }}
      />
    </div>
  );
};

export default ResultsTable; 