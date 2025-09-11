import React, { useState, useEffect } from 'react';
import { getTrainingList } from '../../../api/training.js';
import { uid } from '../../../api/uid.js';
import styles from './TrainingIdSelector.module.css';

const TrainingIdSelector = ({ 
  selectedTid, 
  onTidChange, 
  projectId, 
  showCompletedOnly = true,
  placeholder = "Select Training ID",
  className = "",
  disabled = false 
}) => {
  const [trainings, setTrainings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTrainings = async () => {
      if (!projectId) {
        setTrainings([]);
        return;
      }
      
      setLoading(true);
      setError(null);
      try {
        const result = await getTrainingList({ uid });
        // projectId로 필터링하고, showCompletedOnly가 true면 completed 상태만
        const filteredTrainings = result.filter(training => {
          const projectMatch = training.pid === projectId;
          const statusMatch = showCompletedOnly ? training.status === 'completed' : true;
          return projectMatch && statusMatch;
        });
        setTrainings(filteredTrainings);
      } catch (err) {
        setError(err.message);
        console.error('Failed to fetch training list:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchTrainings();
  }, [projectId, showCompletedOnly]);

  const handleChange = (event) => {
    const selectedValue = event.target.value;
    onTidChange(selectedValue);
  };

  if (!projectId) {
    return (
      <div className={`${styles.container} ${className}`}>
        <select disabled className={styles.select}>
          <option>Please set Project ID first</option>
        </select>
        <small className={styles.noDataText}>
          Project ID를 먼저 설정해주세요.
        </small>
      </div>
    );
  }

  if (loading) {
    return (
      <div className={`${styles.container} ${className}`}>
        <select disabled className={styles.select}>
          <option>Loading...</option>
        </select>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`${styles.container} ${className}`}>
        <select disabled className={`${styles.select} ${styles.error}`}>
          <option>Error loading trainings</option>
        </select>
        <small className={styles.errorText}>{error}</small>
      </div>
    );
  }

  return (
    <div className={`${styles.container} ${className}`}>
      <select
        value={selectedTid}
        onChange={handleChange}
        className={styles.select}
        disabled={disabled}
      >
        <option value="">{placeholder}</option>
        {trainings.map(training => (
          <option key={training.tid} value={training.tid}>
            {training.tid} - {training.origin_tid || training.parameters?.model || 'Unknown'} 
            ({training.origin_dataset_name || 'Unknown Dataset'})
          </option>
        ))}
      </select>
      
      {trainings.length === 0 && !loading && (
        <small className={styles.noDataText}>
          {showCompletedOnly 
            ? 'No completed trainings found for this project.' 
            : 'No trainings found for this project.'}
        </small>
      )}
    </div>
  );
};

export default TrainingIdSelector;
