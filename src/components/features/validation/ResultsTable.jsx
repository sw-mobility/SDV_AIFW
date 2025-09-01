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
              Validation ID
            </th>
            <th style={{ 
              padding: '16px 12px', 
              textAlign: 'left',
              fontWeight: '600',
              fontSize: '14px',
              color: '#374151',
              borderBottom: '2px solid #e2e8f0'
            }}>
              Model
            </th>
            <th style={{ 
              padding: '16px 12px', 
              textAlign: 'left',
              fontWeight: '600',
              fontSize: '14px',
              color: '#374151',
              borderBottom: '2px solid #e2e8f0'
            }}>
              Dataset
            </th>
            <th style={{ 
              padding: '16px 12px', 
              textAlign: 'left',
              fontWeight: '600',
              fontSize: '14px',
              color: '#374151',
              borderBottom: '2px solid #e2e8f0'
            }}>
              Metrics
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
            <th style={{ 
              padding: '16px 12px', 
              textAlign: 'left',
              fontWeight: '600',
              fontSize: '14px',
              color: '#374151',
              borderBottom: '2px solid #e2e8f0'
            }}>
              Timestamp
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
                         <tr key={i} style={{ 
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
                {r.vid || 'N/A'}
              </td>
              <td style={{ 
                padding: '16px 12px',
                fontSize: '14px',
                color: '#374151',
                fontWeight: '500'
              }}>
                {r.model}
              </td>
              <td style={{ 
                padding: '16px 12px',
                fontSize: '14px',
                color: '#374151',
                fontWeight: '500'
              }}>
                {r.dataset}
              </td>
                             <td 
                 style={{ 
                   padding: '16px 12px',
                   fontSize: '13px',
                   color: '#6b7280',
                   cursor: r.metrics && Object.keys(r.metrics).length > 0 ? 'pointer' : 'default'
                 }}
                 onClick={() => handleMetricsClick(r.metrics)}
                 title={r.metrics && Object.keys(r.metrics).length > 0 ? "Click to view detailed metrics" : ""}
               >
                {renderMetrics(r.metrics)}
              </td>
              <td style={{ padding: '16px 12px' }}>
                <span style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: '600',
                  textTransform: 'capitalize',
                  background: r.status === 'completed' ? '#dcfce7' : 
                             r.status === 'failed' ? '#fef2f2' : '#dbeafe',
                  color: r.status === 'completed' ? '#16a34a' : 
                         r.status === 'failed' ? '#dc2626' : '#1d4ed8',
                  border: r.status === 'completed' ? '1px solid #bbf7d0' : 
                          r.status === 'failed' ? '1px solid #fecaca' : '1px solid #bfdbfe'
                }}>
                  {r.status || 'N/A'}
                </span>
              </td>
              <td style={{ 
                padding: '16px 12px',
                fontSize: '13px',
                color: '#6b7280',
                fontWeight: '400'
              }}>
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