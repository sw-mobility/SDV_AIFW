import React from 'react';

/**
 * 기능: 라벨링 결과 테이블
 * 주요 기능: 라벨링 결과를 테이블 형태로 표시 (Validation 스타일 참고)
 * @param result
 * @param dataset
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ResultsTable = ({ result, dataset }) => {
  if (!result) return null;

  const formatTimestamp = () => {
    return new Date().toLocaleString();
  };

  // API 응답에서 정보 추출
  const getResultInfo = () => {
    if (typeof result === 'string') {
      return {
        message: result,
        outputPath: 'runs/labeling/exp',
        processedImages: dataset?.total || 'N/A'
      };
    }
    
    if (result?.data) {
      return {
        message: result.message || 'Labeling completed successfully',
        outputPath: result.data?.output_path || 'runs/labeling/exp',
        processedImages: result.data?.processed_images || dataset?.total || 'N/A',
        detectionCount: result.data?.detection_count || 'N/A'
      };
    }
    
    return {
      message: result?.message || 'Labeling completed successfully',
      outputPath: 'runs/labeling/exp',
      processedImages: dataset?.total || 'N/A'
    };
  };

  const resultInfo = getResultInfo();

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 16 }}>
        <thead>
          <tr style={{ background: '#f8fafc' }}>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Labeling ID</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Model</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Dataset</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Output</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Status</th>
            <th style={{ padding: '10px', borderBottom: '1px solid #e2e8f0', textAlign: 'left' }}>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
            <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '12px' }}>
              labeling_{Date.now()}
            </td>
            <td style={{ padding: '10px' }}>YOLO11n</td>
            <td style={{ padding: '10px' }}>
              <div>
                <div style={{ fontWeight: '500' }}>{dataset?.name || 'N/A'}</div>
                <div style={{ fontSize: '12px', color: '#6b7280' }}>
                  {resultInfo.processedImages} images
                </div>
              </div>
            </td>
            <td style={{ padding: '10px', fontSize: '12px', color: '#6b7280' }}>
              <div>
                <div>{resultInfo.outputPath}</div>
                {resultInfo.detectionCount !== 'N/A' && (
                  <div style={{ fontSize: '11px', color: '#9ca3af' }}>
                    {resultInfo.detectionCount} detections
                  </div>
                )}
              </div>
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
      
      {/* 추가 정보 섹션 */}
      <div style={{ 
        marginTop: 16, 
        padding: 16, 
        background: '#f8fafc', 
        borderRadius: 8, 
        border: '1px solid #e2e8f0' 
      }}>
        <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
          Labeling Summary
        </h4>
        <div style={{ fontSize: '13px', color: '#6b7280', lineHeight: 1.5 }}>
          {resultInfo.message}
        </div>
      </div>
    </div>
  );
};

export default ResultsTable;
