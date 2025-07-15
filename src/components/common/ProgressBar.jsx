import React from 'react';
import styles from './ProgressBar.module.css';

export default function ProgressBar({ label, percentage }) {
    return (
        <div className={styles['progress-bar-container']}>
            {label && <div className={styles['progress-bar-label']}>{label}</div>}
            <div className={styles['progress-bar-bg']}>
                <div className={styles['progress-bar-fill']} style={{ width: `${percentage}%` }}></div>
            </div>
            <div className={styles['progress-bar-text']}>{percentage}% complete</div>
        </div>
    );
}
