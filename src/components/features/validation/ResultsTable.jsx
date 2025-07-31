import React from 'react';

const ResultsTable = ({ results }) => {
  if (results.length === 0) return null;

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
      <thead>
        <tr style={{ background: '#f8fafc' }}>
          <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
          <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Dataset</th>
          <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Metric</th>
          <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Value</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r, i) => (
          <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
            <td style={{ padding: '10px' }}>{r.model}</td>
            <td style={{ padding: '10px' }}>{r.dataset}</td>
            <td style={{ padding: '10px' }}>{r.metric}</td>
            <td style={{ padding: '10px' }}>{r.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default ResultsTable; 