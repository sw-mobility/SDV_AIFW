import React from 'react';
import { Code } from 'lucide-react';
import styles from './ExpertModeToggle.module.css';

const ExpertModeToggle = ({ isActive, onToggle }) => {
  return (
    <div className={styles.expertModeToggle}>
      <button
        className={`${styles.expertModeButton} ${isActive ? styles.active : ''}`}
        onClick={onToggle}
      >
        <Code size={16} />
        {isActive ? 'Expert Mode Active' : 'Enable Expert Mode'}
        {isActive && <span className={styles.expertModeBadge}>ON</span>}
      </button>
      <div className={styles.expertModeDescription}>
        <span>Expert mode allows you to edit training code directly</span>
      </div>
    </div>
  );
};

export default ExpertModeToggle; 