import React from 'react';

/**
 * 기능: 검증 결과 테이블
 * 주요 기능: 검증 결과를 테이블 형태로 표시
 * @param results
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ResultsTable = ({ results }) => {
  if (results.length === 0) return null;

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  const renderResults = (results) => {
    if (!results || typeof results !== 'object') return 'N/A';
    
    const resultItems = [];
    for (const [key, value] of Object.entries(results)) {
      if (typeof value === 'number') {
        resultItems.push(`${key}: ${value.toFixed(4)}`);
      } else {
        resultItems.push(`${key}: ${value}`);
      }
    }
    
    return resultItems.length > 0 ? resultItems.join(', ') : 'N/A';
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
        <thead>
          <tr style={{ background: '#f8fafc' }}>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Validation ID</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Dataset</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Results</th>
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
              <td style={{ padding: '10px', fontSize: '12px', color: '#6b7280' }}>
                {renderResults(r.results)}
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
    </div>
  );
};

export default ResultsTable; 