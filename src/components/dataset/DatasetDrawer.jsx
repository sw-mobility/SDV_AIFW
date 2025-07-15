import React from 'react';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import { Download, Trash2, Upload, Database, Tag, Calendar } from 'lucide-react';
import { useDatasetContext } from '../../context/DatasetContext';
import StatusChip from '../common/StatusChip.jsx';
import Loading from '../common/Loading.jsx';
import ErrorMessage from '../common/ErrorMessage.jsx';
import EmptyState from '../common/EmptyState.jsx';
import styles from './DatasetDrawer.module.css';
import { downloadDataset, deleteDataset } from '../../api/datasets.js';
import DatasetUploadModal from './DatasetUploadModal.jsx';

const DatasetCard = ({ dataset, onDownload, onDelete }) => (
    <div className={styles['dataset-card']}>
        <div className={styles['dataset-card-header']}>
            <StatusChip status={dataset.status} />
            <div className={styles['dataset-card-actions']}>
                <IconButton size="small" title="Download" onClick={() => onDownload(dataset)}>
                    <Download size={16} />
                </IconButton>
                <IconButton size="small" title="Delete" onClick={() => onDelete(dataset)}>
                    <Trash2 size={16} />
                </IconButton>
            </div>
        </div>
        <div className={styles['dataset-card-name']}>
            {dataset.name}
        </div>
        <div className={styles['dataset-card-info']}>
            {dataset.type === 'Image' ? <Database size={14} /> : <Tag size={14} />}
            <span>{dataset.type}</span>
            <span>· {dataset.size}</span>
        </div>
        <div className={styles['dataset-card-date']}>
            <Calendar size={13} />
            <span>{dataset.lastModified}</span>
        </div>
    </div>
);

const DatasetDrawer = ({ open, onClose }) => {
    const { datasets, loading, error, reload } = useDatasetContext();
    const [uploadOpen, setUploadOpen] = React.useState(false);

    const handleDownload = async (dataset) => {
        try {
            await downloadDataset(dataset.id, dataset.type === 'Image' ? 'raw' : 'labeled');
            console.log('Download started for:', dataset.name);
        } catch (error) {
            console.error('Download failed:', error.message);
            // TODO: Show error notification
        }
    };

    const handleDelete = async (dataset) => {
        try {
            await deleteDataset(dataset.id, dataset.type === 'Image' ? 'raw' : 'labeled');
            console.log('Dataset deleted:', dataset.name);
            // Refresh the dataset list
            reload();
        } catch (error) {
            console.error('Delete failed:', error.message);
            // TODO: Show error notification
        }
    };

    return (
        <Drawer 
            anchor="right" 
            open={open} 
            onClose={onClose} 
            PaperProps={{ sx: { width: 340, boxShadow: 3 } }}
        >
            <div className={styles['drawer-header']}>
                <div className={styles['drawer-title']}>
                    Data Management
                </div>
                <IconButton onClick={onClose} size="small">
                    <span style={{ fontSize: 24, fontWeight: 700 }}>&times;</span>
                </IconButton>
            </div>
            <Divider />
            <div className={styles['drawer-content']}>
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
                    onClick={() => setUploadOpen(true)}
                >
                    Upload Dataset
                </Button>
                <DatasetUploadModal isOpen={uploadOpen} onClose={() => setUploadOpen(false)} />
                
                {loading && <Loading />}
                {error && <ErrorMessage message={error.message} />}
                
                <div className={styles['dataset-list']}>
                    {datasets.map(dataset => (
                        <DatasetCard
                            key={dataset.id}
                            dataset={dataset}
                            onDownload={handleDownload}
                            onDelete={handleDelete}
                        />
                    ))}
                </div>
                
                {datasets.length === 0 && !loading && (
                    <EmptyState message="No datasets found." />
                )}
            </div>
        </Drawer>
    );
};

export default DatasetDrawer;