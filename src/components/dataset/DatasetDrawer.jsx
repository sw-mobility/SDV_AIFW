import React from 'react';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import { Download, Trash2, Upload, Database, Tag, Calendar } from 'lucide-react';
import { useDatasetContext } from '../../context/DatasetContext';
import Chip from '@mui/material/Chip';

const getStatusColor = (status) => {
    switch (status) {
        case 'Active': return 'success';
        case 'Training': return 'warning';
        case 'Deployed': return 'info';
        case 'Processing': return 'warning';
        default: return 'default';
    }
};

const getStatusText = (status) => {
    switch (status) {
        case 'Active': return 'active';
        case 'Training': return 'training';
        case 'Deployed': return 'deployed';
        case 'Processing': return 'processing';
        default: return status;
    }
};

const DatasetDrawer = ({ open, onClose }) => {
    const { datasets, loading, error } = useDatasetContext();

    return (
        <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: 340, boxShadow: 3 } }}>
            <div style={{ display: 'flex', alignItems: 'center', padding: '1.2rem 1.5rem', justifyContent: 'space-between' }}>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>Data Management</Typography>
                <IconButton onClick={onClose} size="small">
                    <span style={{ fontSize: 24, fontWeight: 700 }}>&times;</span>
                </IconButton>
            </div>
            <Divider />
            <div style={{padding: '1.5rem' }}>
                <Button
                    variant="contained"
                    startIcon={<Upload size={16} />}
                    fullWidth
                    sx={{
                        mb: 2,
                        borderRadius: '12px',
                        fontWeight: 600,
                        textTransform: 'none'
                    }}
                >
                    Upload Dataset
                </Button>
                {loading && <Typography variant="body2">Loading...</Typography>}
                {error && <Typography color="error">{error.message}</Typography>}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
                    {datasets.map(ds => (
                        <div
                            key={ds.id}
                            style={{
                                border: '1px solid var(--color-border-primary, #e0e0e0)',
                                borderRadius: 16,
                                padding: '1rem',
                                background: '#fff',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.03)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '0.5rem'
                            }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <Chip
                                    label={getStatusText(ds.status)}
                                    color={getStatusColor(ds.status)}
                                    size="small"
                                    variant="outlined"
                                    sx={{ fontSize: 11, height: 20 }}
                                />
                                <div style={{ flex: 1 }} />
                                <IconButton size="small" title="Download">
                                    <Download size={16} />
                                </IconButton>
                                <IconButton size="small" title="Delete">
                                    <Trash2 size={16} />
                                </IconButton>
                            </div>
                            <div style={{ fontWeight: 600, fontSize: 16, color: '#222', marginTop: 2 }}>
                                {ds.name}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#666', fontSize: 13 }}>
                                {ds.type === 'Image' ? <Database size={14} /> : <Tag size={14} />}
                                <span>{ds.type}</span>
                                <span>· {ds.size}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#888', fontSize: 12 }}>
                                <Calendar size={13} />
                                <span>{ds.lastModified}</span>
                            </div>
                        </div>
                    ))}
                </div>
                {datasets.length === 0 && !loading && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>No datasets found.</Typography>
                )}
            </div>
        </Drawer>
    );
};

export default DatasetDrawer;