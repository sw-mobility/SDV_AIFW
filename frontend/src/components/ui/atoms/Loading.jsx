import React from 'react';
import CircularProgress from '@mui/material/CircularProgress';

export default function Loading({ style, fullHeight = false }) {
    const containerStyle = {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
        height: fullHeight ? '100%' : '200px',
        minHeight: fullHeight ? '400px' : '200px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <CircularProgress />
        </div>
    );
} 