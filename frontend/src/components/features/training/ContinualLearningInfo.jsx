import React from 'react';
import styles from './ContinualLearningInfo.module.css';

const ContinualLearningInfo = () => {
  return (
    <div className={styles.infoCard}>
      <b>Continual Learning</b> allows you to update a model incrementally with new data, starting from a previous snapshot. Select a base snapshot and a new dataset to continue training.
    </div>
  );
};

export default ContinualLearningInfo; 