import React from 'react';
import './SectionTitle.module.css';

export default function SectionTitle({ children, size = 'lg' }) {
    return (
        <h2 className={`section-title section-title-${size}`}>
            {children}
        </h2>
    );
}
