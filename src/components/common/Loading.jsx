import React from 'react';
import CircularProgress from '@mui/material/CircularProgress';

export default function Loading({ style }) {
    return (
        <div style={{ padding: '2rem', textAlign: 'center', ...style }}>
            <CircularProgress />
        </div>
    );
} 