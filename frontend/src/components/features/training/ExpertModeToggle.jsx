import React from 'react';
import { Code } from 'lucide-react';
import styles from './ExpertModeToggle.module.css';

/**
 * 전문가 모드 토글
 * 주요 기능: 고급 파라미터 설정 모드 전환
 * @param isActive
 * @param onToggle
 * @returns {Element}
 * @constructor
 */
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