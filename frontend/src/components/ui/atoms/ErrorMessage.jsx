import React from 'react';

export default function ErrorMessage({ message, style, fullHeight = false }) {
    const containerStyle = {
        color: 'red',
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
            Error: {message}
        </div>
    );
} 