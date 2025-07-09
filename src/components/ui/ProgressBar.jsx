import React from 'react';
import './ProgressBar.module.css';

export default function ProgressBar({ label, percentage }) {
    return (
        <div className="progress-bar-container">
            <div className="progress-bar-label">{label}</div>
            <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${percentage}%` }}></div>
            </div>
            <div className="progress-bar-text">{percentage}% complete</div>
        </div>
    );
}
