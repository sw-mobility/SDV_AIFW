import React from 'react';
import styles from './Card.module.css';

export default function Card({
    children,
    className = '',
    onClick,
    ...props
}) {
    const cardClass = [
        styles.card,
        className
    ].filter(Boolean).join(' ');

    const handleClick = (e) => {
        if (onClick) {
            onClick(e);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick(e);
        }
    };

    return (
        <div 
            className={cardClass}
            onClick={handleClick}
            onKeyDown={handleKeyDown}
            tabIndex={onClick ? 0 : undefined}
            role={onClick ? 'button' : undefined}
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
    return (
        <div 
            className={`${styles.cardGrid} ${className}`}
            data-columns={columns}
            style={{ gap }}
        >
            {children}
        </div>
    );
}