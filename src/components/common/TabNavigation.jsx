import React from 'react';
import styles from './TabNavigation.module.css';

const TabNavigation = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className={styles.tabNavigation}>
      {tabs.map((tab) => (
        <button
          key={tab.value}
          className={`${styles.tabButton} ${activeTab === tab.value ? styles.activeTab : ''}`}
          onClick={() => onTabChange(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
};

export default TabNavigation; 