import React from 'react';
import Chip from '@mui/material/Chip';
import { getStatusColor, getStatusText } from '../../../domain/project/status.js';

export default function StatusChip({ status, className }) {
    return (
        <Chip
            label={getStatusText(status)}
            color={getStatusColor(status)}
            size="small"
            variant="outlined"
            className={className}
        />
    );
} 