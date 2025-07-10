import React from 'react';
import styles from './Card.module.css';
// 추후 link to 넣기

export default function Card({
    children,
    className = '',
    layout = 'default', // 'default', 'compact', 'wide'
    onClick,
    hoverable = false,
    selected = false,
    disabled = false,
    ...props
}) {
    const cardClass = [
        styles.card,
        styles[layout],
        hoverable && styles.hoverable,
        selected && styles.selected,
        disabled && styles.disabled,
        className
    ].filter(Boolean).join(' ');

    const handleClick = (e) => {
        if (onClick && !disabled) {
            onClick(e);
        }
    };

    return (
        <div 
            className={cardClass}
            onClick={handleClick}
            {...props}
        >
            {children}
        </div>
    );
}

export function CardGrid({ 
    children, 
    columns = 2,
    gap = '1rem',
    className = '' 
}) {
    const gridStyle = {
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: gap,
    };

    return (
        <div className={`${styles.cardGrid} ${className}`} style={gridStyle}>
            {children}
        </div>
    );
}