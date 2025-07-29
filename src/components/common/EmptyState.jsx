import React from 'react';

export default function EmptyState({ message, style, fullHeight = false }) {
    const containerStyle = {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
        height: fullHeight ? '100%' : '200px',
        minHeight: fullHeight ? '400px' : '200px',
        textAlign: 'center',
        ...style
    };

    return (
        <div style={containerStyle}>
            {message}
        </div>
    );
} 