import React, { useState } from 'react';
import DatasetDataPanel from '../dataset/DatasetDataPanel';
import { uid } from '../../../api/uid.js';

/**
 * 기능: Labeled Datasets 목록 표시
 * 주요 기능: 실시간으로 업데이트되는 labeled datasets 목록을 표시
 * @param labeledDatasets
 * @param isPolling
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const LabeledDatasetsList = ({ labeledDatasets, isPolling }) => {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [isDataPanelOpen, setIsDataPanelOpen] = useState(false);

  const handleDatasetClick = (dataset) => {
    // uid를 포함한 dataset 객체 생성
    const datasetWithUid = {
      ...dataset,
      uid: uid,
      _id: dataset._id || dataset.id || dataset.did,
      datasetType: 'labeled'
    };
    setSelectedDataset(datasetWithUid);
    setIsDataPanelOpen(true);
  };

  const closeDataPanel = () => {
    setIsDataPanelOpen(false);
    setSelectedDataset(null);
  };
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: 16 
      }}>
        <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#1e293b' }}>
          Labeled Datasets
        </h3>
        {isPolling && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 8,
            padding: '4px 12px',
            background: '#dbeafe',
            borderRadius: '16px',
            fontSize: '12px',
            color: '#1d4ed8'
          }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#1d4ed8',
              animation: 'pulse 1.5s infinite'
            }}></div>
            Polling for new datasets...
          </div>
        )}
      </div>

      {labeledDatasets.length === 0 ? (
        <div style={{
          padding: 32,
          textAlign: 'center',
          background: '#f8fafc',
          borderRadius: 8,
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: 8 }}>
            No labeled datasets yet
          </div>
          <div style={{ fontSize: '12px', color: '#9ca3af' }}>
            Run labeling to create your first labeled dataset
          </div>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                         <thead>
               <tr style={{ background: '#f8fafc' }}>
                 <th style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', textAlign: 'left', fontSize: '13px', fontWeight: '500', color: '#6b7280' }}>Name</th>
                 <th style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', textAlign: 'left', fontSize: '13px', fontWeight: '500', color: '#6b7280' }}>Type</th>
                 <th style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', textAlign: 'left', fontSize: '13px', fontWeight: '500', color: '#6b7280' }}>Size</th>
                 <th style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', textAlign: 'left', fontSize: '13px', fontWeight: '500', color: '#6b7280' }}>Created</th>
               </tr>
             </thead>
            <tbody>
                             {labeledDatasets.map((dataset, index) => (
                 <tr 
                   key={dataset._id || dataset.id || index} 
                   style={{ 
                     borderBottom: '1px solid #f1f5f9',
                     cursor: 'pointer',
                     transition: 'background-color 0.2s ease'
                   }}
                   onClick={() => handleDatasetClick(dataset)}
                   onMouseEnter={(e) => e.target.closest('tr').style.backgroundColor = '#f8fafc'}
                   onMouseLeave={(e) => e.target.closest('tr').style.backgroundColor = 'transparent'}
                 >
                   <td style={{ padding: '12px', fontWeight: '500' }}>
                     {dataset.name || 'Unnamed Dataset'}
                   </td>
                  <td style={{ padding: '12px' }}>
                    <span style={{
                      padding: '2px 8px',
                      borderRadius: '12px',
                      fontSize: '11px',
                      fontWeight: '500',
                      background: '#dcfce7',
                      color: '#166534'
                    }}>
                      {dataset.type || dataset.datasetType || 'Labeled'}
                    </span>
                  </td>
                  <td style={{ padding: '12px', fontSize: '12px', color: '#6b7280' }}>
                    {dataset.total || dataset.size || 'N/A'} items
                  </td>
                                     <td style={{ padding: '12px', fontSize: '12px', color: '#6b7280' }}>
                     {formatTimestamp(dataset.created_at || dataset.createdAt)}
                   </td>
                 </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

             <style jsx>{`
         @keyframes pulse {
           0%, 100% { opacity: 1; }
           50% { opacity: 0.5; }
         }
       `}</style>

       {/* Dataset Detail Panel */}
       <DatasetDataPanel
         open={isDataPanelOpen}
         onClose={closeDataPanel}
         dataset={selectedDataset}
       />
     </div>
   );
 };

export default LabeledDatasetsList;
