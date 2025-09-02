import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import Table from '../../ui/atoms/Table.jsx';
import styles from './TrainingResultList.module.css';

const TrainingResultList = () => {
  const [trainings, setTrainings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTrainings = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getTrainingList({ uid });
      console.log('Raw training list:', result);
      
      if (result && Array.isArray(result)) {
        // 최근 생성된 training이 가장 위에 오도록 정렬
        const sortedTrainings = result.sort((a, b) => {
          const dateA = new Date(a.started_at || a.created_at || 0);
          const dateB = new Date(b.started_at || b.created_at || 0);
          return dateB - dateA; // 내림차순 (최신순)
        });
        
        console.log('Sorted trainings:', sortedTrainings);
        setTrainings(sortedTrainings);
      } else {
        console.warn('Training list is not an array:', result);
        setTrainings([]);
      }
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch training list:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrainings();
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

  const getAlgorithmDisplay = (training) => {
    // origin_tid가 있으면 custom model, 없으면 pretrained model
    if (training.origin_tid && training.origin_tid !== training.parameters?.model) {
      return `Custom (${training.origin_tid})`;
    }
    return training.parameters?.model || training.origin_tid || '-';
  };

  const getDatasetDisplay = (training) => {
    if (training.origin_dataset_name) {
      return training.origin_dataset_name;
    }
    if (training.origin_did) {
      return training.origin_did;
    }
    return '-';
  };

  const columns = [
    { 
      key: 'tid', 
      label: 'Training ID', 
      sortable: true,
      width: '120px'
    },
    { 
      key: 'algorithm', 
      label: 'Algorithm', 
      sortable: true,
      width: '140px'
    },
    { 
      key: 'dataset', 
      label: 'Dataset', 
      sortable: true,
      width: '160px'
    },
    { 
      key: 'status', 
      label: 'Status', 
      sortable: true,
      width: '100px'
    },
    { 
      key: 'started_at', 
      label: 'Started At', 
      sortable: true,
      width: '140px'
    },
    { 
      key: 'epochs', 
      label: 'Epochs', 
      sortable: true,
      width: '80px'
    },
    { 
      key: 'actions', 
      label: 'Actions', 
      sortable: false,
      width: '120px'
    }
  ];

  const tableData = trainings.map(training => ({
    id: training.tid, // Table 컴포넌트의 rowKey로 사용
    cells: [
      (
        <div className={styles.tidCell}>
          <span className={styles.tidValue}>{training.tid}</span>
          {training.pid && <span className={styles.projectId}>P: {training.pid}</span>}
        </div>
      ),
      (
        <div className={styles.algorithmCell}>
          <span className={styles.algorithmName}>{getAlgorithmDisplay(training)}</span>
          {training.parameters?.pretrained !== false && (
            <span className={styles.pretrainedTag}>Pretrained</span>
          )}
        </div>
      ),
      (
        <div className={styles.datasetCell}>
          <span className={styles.datasetName}>{getDatasetDisplay(training)}</span>
          {training.classes && training.classes.length > 0 && (
            <span className={styles.classesCount}>
              {training.classes.length} classes
            </span>
          )}
        </div>
      ),
      getStatusBadge(training.status),
      formatDate(training.started_at),
      (
        <div className={styles.epochsCell}>
          <span className={styles.epochsValue}>
            {training.parameters?.epochs || '-'}
          </span>
          {training.parameters?.batch && (
            <span className={styles.batchInfo}>Batch: {training.parameters.batch}</span>
          )}
        </div>
      ),
      (
        <div className={styles.actionButtons}>
          <button 
            className={`${styles.actionBtn} ${styles.viewBtn}`}
            onClick={() => console.log('View details for:', training.tid)}
            title="View Details"
          >
            View
          </button>
          <button 
            className={`${styles.actionBtn} ${styles.downloadBtn}`}
            onClick={() => console.log('Download model for:', training.tid)}
            title="Download Model"
          >
            Download
          </button>
          <button 
            className={`${styles.actionBtn} ${styles.cloneBtn}`}
            onClick={() => console.log('Clone training for:', training.tid)}
            title="Clone Training"
          >
            Clone
          </button>
        </div>
      )
    ]
  }));

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading training results...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.errorMessage}>Error: {error}</p>
        <button onClick={fetchTrainings} className={styles.retryBtn}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.title}>Training Results</h3>
          <span className={styles.count}>({trainings.length})</span>
        </div>
        <div className={styles.headerRight}>
          <button onClick={fetchTrainings} className={styles.refreshBtn}>
            REFRESH
          </button>
        </div>
      </div>
      
      {trainings.length === 0 ? (
        <div className={styles.emptyState}>
          <h4>No training results found</h4>
          <p>Start a new training to see results here.</p>
        </div>
      ) : (
        <div className={styles.tableContainer}>
          <Table 
            data={tableData} 
            columns={columns}
            className={styles.trainingTable}
            rowKey="id"
          />
        </div>
      )}
    </div>
  );
};

export default TrainingResultList;
