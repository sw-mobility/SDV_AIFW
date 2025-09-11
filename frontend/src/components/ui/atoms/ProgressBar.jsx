import React from 'react';
import styles from './ProgressBar.module.css';
import { Loader, CheckCircle, XCircle } from 'lucide-react';

export default function ProgressBar({ 
    label, 
    percentage, 
    status = 'idle', 
    completeText = 'Completed!',
    runningText = 'Processing...',
    errorText = 'Failed.',
    idleText = 'Ready.'
}) {
    // status: idle | running | success | error
    let barColor = '#4f8cff';
    let text = `${percentage}% complete`;
    let icon = null;
    let textColor = '#64748b';
    if (status === 'running') {
        text = runningText;
        icon = <Loader size={16} className={styles.spinner} />;
    } else if (status === 'success') {
        barColor = '#22c55e';
        text = completeText;
        icon = <CheckCircle size={16} color="#22c55e" />;
        textColor = '#22c55e';
    } else if (status === 'error') {
        barColor = '#ef4444';
        text = errorText;
        icon = <XCircle size={16} color="#ef4444" />;
        textColor = '#ef4444';
    } else if (status === 'idle') {
        text = idleText;
    }
    return (
        <div className={styles['progress-bar-container']}>
            {label && <div className={styles['progress-bar-label']}>{label}</div>}
            <div className={styles['progress-bar-bg']}>
                <div className={styles['progress-bar-fill']} style={{ width: `${percentage}%`, background: barColor }}></div>
            </div>
            <div className={styles['progress-bar-text']} style={{ color: textColor, display: 'flex', alignItems: 'center', gap: 6 }}>
                {icon}
                {text}
            </div>
        </div>
    );
}
