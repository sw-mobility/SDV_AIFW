import React from 'react';

export default function ErrorMessage({ message, style }) {
    return (
        <div style={{ color: 'red', padding: '2rem', textAlign: 'center', ...style }}>
            Error: {message}
        </div>
    );
} 