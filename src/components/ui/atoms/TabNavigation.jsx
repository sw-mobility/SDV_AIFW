import React from 'react';
import styles from './TabNavigation.module.css';
import { TRAINING_TYPES } from '../../../domain/training/trainingTypes.js';

const TabNavigation = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className={styles.tabNavigation}>
      {tabs.map((tab) => {
        const isContinualTraining = tab.value === TRAINING_TYPES.CONTINUAL;
        const isDisabled = isContinualTraining;
        
        return (
          <button
            key={tab.value}
            className={`${styles.tabButton} ${activeTab === tab.value ? styles.activeTab : ''} ${isDisabled ? styles.disabledTab : ''}`}
            onClick={() => !isDisabled && onTabChange(tab.value)}
            disabled={isDisabled}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
};

export default TabNavigation; 