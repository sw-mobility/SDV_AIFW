import React from 'react';

export default function EmptyState({ message, style }) {
    return (
        <div style={{ padding: '2rem', textAlign: 'center', ...style }}>
            {message}
        </div>
    );
} 