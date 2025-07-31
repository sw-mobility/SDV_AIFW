import React from 'react';
import styles from './SectionTitle.module.css';

export default function SectionTitle({ children, size = 'lg' }) {
    return (
        <h2 className={`${styles['section-title']} ${styles[`section-title-${size}`]}`}>
            {children}
        </h2>
    );
}
