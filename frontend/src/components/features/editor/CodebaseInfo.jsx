import React from 'react';
import { CircularProgress, Box } from '@mui/material';
import styles from './CodebaseInfo.module.css';

/**
 * 선택된 코드베이스의 정보를 표시하는 컴포넌트
 */
const CodebaseInfo = ({ codebase, loading = false, files = {}, lastSavedAt = null }) => {
  if (loading) {
    return (
      <div className={styles.codebaseInfo}>
        <Box 
          display="flex" 
          justifyContent="center" 
          alignItems="center" 
          minHeight="120px"
        >
          <CircularProgress size={24} />
        </Box>
      </div>
    );
  }

  if (!codebase) {
    return (
      <div className={styles.codebaseInfo}>
        <div className={styles.emptyState}>
          <p>Select a codebase</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.codebaseInfo}>
      <div className={styles.header}>
        <div className={styles.title}>
          {codebase.name || codebase.cid}
        </div>
        <div className={styles.badge}>
          {codebase.algorithm || 'Unknown'}
        </div>
      </div>

      <div className={styles.metaInfo}>
        <div className={styles.metaItem}>
          <span className={styles.label}>Stage:</span>
          <span className={styles.value}>{codebase.stage || 'N/A'}</span>
        </div>

        <div className={styles.metaItem}>
          <span className={styles.label}>Task Type:</span>
          <span className={styles.value}>{codebase.task_type || 'N/A'}</span>
        </div>

        <div className={styles.metaItem}>
          <span className={styles.label}>Algorithm:</span>
          <span className={styles.value}>{codebase.algorithm || 'N/A'}</span>
        </div>

        <div className={styles.metaItem}>
          <span className={styles.label}>Created:</span>
          <span className={styles.value}>
            {codebase.created_at ? new Date(codebase.created_at).toLocaleDateString() : 'N/A'}
          </span>
        </div>
      </div>

      {codebase.description && (
        <div className={styles.description}>
          <h4>Description</h4>
          <p>{codebase.description}</p>
        </div>
      )}

      <div className={styles.stats}>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Files</span>
          <span className={styles.statValue}>{Object.keys(files).length}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Last Modified</span>
          <span className={styles.statValue}>
            {lastSavedAt ? new Date(lastSavedAt).toLocaleDateString() : 'N/A'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default CodebaseInfo;
