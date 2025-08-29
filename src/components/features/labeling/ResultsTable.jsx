import React from 'react';

/**
 * 기능: 라벨링 결과 테이블
 * 주요 기능: 라벨링 결과를 테이블 형태로 표시
 * @param result
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ResultsTable = ({ result }) => {
  if (!result) return null;

  const formatTimestamp = () => {
    return new Date().toLocaleString();
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
        <thead>
          <tr style={{ background: '#f8fafc' }}>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Labeling ID</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Result</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Status</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
            <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '12px' }}>
              labeling_{Date.now()}
            </td>
            <td style={{ padding: '10px' }}>YOLO Detection</td>
            <td style={{ padding: '10px', fontSize: '12px', color: '#6b7280' }}>
              {result}
            </td>
            <td style={{ padding: '10px' }}>
              <span style={{
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: '500',
                background: '#dcfce7',
                color: '#16a34a'
              }}>
                completed
              </span>
            </td>
            <td style={{ padding: '10px', fontSize: '12px', color: '#6b7280' }}>
              {formatTimestamp()}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
