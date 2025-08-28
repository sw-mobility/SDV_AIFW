import React, { useState } from 'react';
import MetricsDetailModal from './MetricsDetailModal.jsx';

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
    
         // 주요 metrics 표시
     if (metrics['mAP_0.5'] !== undefined || metrics.mAP_0_5 !== undefined) {
       const map05 = metrics['mAP_0.5'] || metrics.mAP_0_5;
       metricItems.push(`mAP@0.5: ${map05.toFixed(4)}`);
     }
     if (metrics['mAP_0.5:0.95'] !== undefined || metrics.mAP_0_5_0_95 !== undefined) {
       const map0595 = metrics['mAP_0.5:0.95'] || metrics.mAP_0_5_0_95;
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
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
        <thead>
          <tr style={{ background: '#f8fafc' }}>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Validation ID</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Dataset</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Metrics</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Status</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '12px' }}>
                {r.vid || 'N/A'}
              </td>
              <td style={{ padding: '10px' }}>{r.model}</td>
              <td style={{ padding: '10px' }}>{r.dataset}</td>
              <td 
                style={{ 
                  padding: '10px', 
                  fontSize: '12px', 
                  color: '#6b7280',
                  cursor: r.metrics && Object.keys(r.metrics).length > 0 ? 'pointer' : 'default'
                }}
                onClick={() => handleMetricsClick(r.metrics)}
                title={r.metrics && Object.keys(r.metrics).length > 0 ? "Click to view detailed metrics" : ""}
              >
                {renderMetrics(r.metrics)}
              </td>
              <td style={{ padding: '10px' }}>
                <span style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: '500',
                  background: r.status === 'completed' ? '#dcfce7' : 
                             r.status === 'failed' ? '#fef2f2' : '#dbeafe',
                  color: r.status === 'completed' ? '#16a34a' : 
                         r.status === 'failed' ? '#dc2626' : '#1d4ed8'
                }}>
                  {r.status || 'N/A'}
                </span>
              </td>
              <td style={{ padding: '10px', fontSize: '12px', color: '#6b7280' }}>
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