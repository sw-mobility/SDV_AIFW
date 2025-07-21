import React from 'react';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import {
    Download,
    Trash2,
    Database,
    Tag,
    Calendar,
    PlusCircle,
    Edit2,
    Image as ImageIcon,
    FileText,
    AudioLines,
    Video
} from 'lucide-react';
import {useDatasetContext} from '../../context/DatasetContext.jsx';
import Loading from '../common/Loading.jsx';
import ErrorMessage from '../common/ErrorMessage.jsx';
import EmptyState from '../common/EmptyState.jsx';
import styles from './Dataset.module.css';
import {downloadDataset, deleteDataset, updateRawDataset} from '../../api/datasets.js';
import DatasetUploadModal from './DatasetUploadModal.jsx';
import {Label} from "recharts";

const DatasetEditModal = ({open, onClose, dataset, onUpdated}) => {
    const [name, setName] = React.useState(dataset?.name || '');
    const [type, setType] = React.useState(dataset?.type || 'Image');
    const [description, setDescription] = React.useState(dataset?.description || '');
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState(null);
    React.useEffect(() => {
        if (open) {
            setName(dataset?.name || '');
            setType(dataset?.type || 'Image');
            setDescription(dataset?.description || '');
            setError(null);
        }
    }, [open, dataset]);
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            await updateRawDataset({
                did: dataset.did || dataset._id || dataset.id,
                name,
                description,
                type
            });
            onUpdated && onUpdated();
            onClose();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    if (!open) return null;
    return (
        <div className={styles['modal-backdrop']}>
            <div className={styles['modal']}> {/* Reuse modal styles */}
                <form onSubmit={handleSubmit} className={styles.formGroup}>
                    <div style={{fontWeight: 600, fontSize: 18, marginBottom: 12}}>Edit Dataset</div>
                    <label className={styles.label}>
                        Name
                        <input type="text" value={name} onChange={e => setName(e.target.value)}
                               className={styles.input}/>
                    </label>
                    <label className={styles.label}>
                        Type
                        <select value={type} onChange={e => setType(e.target.value)} className={styles.input}>
                            {['Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'].map(t => (
                                <option key={t} value={t}>{t}</option>
                            ))}
                        </select>
                    </label>
                    <label className={styles.label}>
                        Description
                        <textarea value={description} onChange={e => setDescription(e.target.value)}
                                  className={styles.input} rows={3} style={{resize: 'vertical'}}/>
                    </label>
                    {error && <div className={styles.fileError}>{error}</div>}
                    <div className={styles.modalActions}>
                        <button type="button" onClick={onClose} className={styles.cancelButton}
                                disabled={loading}>Cancel
                        </button>
                        <Button type="submit" variant="primary" size="medium" disabled={loading || !name}>Save</Button>
                    </div>
                </form>
            </div>
        </div>
    );
};

const typeIconMap = {
    Image: <ImageIcon size={14}/>,
    Text: <FileText size={14}/>,
    Audio: <AudioLines size={14}/>,
    Video: <Video size={14}/>,
    Raw: <Database size={14} style={{"color": "#3498db"}}/>,
    Graph: <Tag size={14}/>,
    Label: <Tag size={14} style={{"color": "#cc305a"}}/>,
};

const DatasetCard = ({dataset, onDownload, onDelete, onEdit}) => (
    <div className={styles['dataset-card']}>
        <div className={styles['dataset-card-header']}>
            <div>
                {dataset.datasetType === 'raw' ? typeIconMap.Raw : typeIconMap.Label}
            </div>
            <div className={styles['dataset-card-actions']}>
                <IconButton size="small" title="Edit" onClick={() => onEdit(dataset)}>
                    <Edit2 size={16}/>
                </IconButton>
                <IconButton size="small" title="Download" onClick={() => onDownload(dataset)}>
                    <Download size={16}/>
                </IconButton>
                <IconButton size="small" title="Delete" onClick={() => onDelete(dataset)}>
                    <Trash2 size={16}/>
                </IconButton>
            </div>
        </div>
        <div className={styles['dataset-card-name']}>
            {dataset.name}
        </div>
        <div className={styles['dataset-card-description']}>
            {dataset.description || <span style={{color: '#bbb'}}>No description</span>}
        </div>
        <div className={styles['dataset-card-info']}>
            {typeIconMap[dataset.type] || <Database size={14}/>}
            <span>{dataset.type}</span>
            {dataset.task_type && <span className={styles['drawer-chip']} title="Task Type">{dataset.task_type}</span>}
            {dataset.label_format &&
                <span className={styles['drawer-chip']} title="Label Format">{dataset.label_format}</span>}
            {dataset.total !== undefined &&
                <span className={styles['drawer-chip']} title="Total">총 {dataset.total}개</span>}
        </div>
        <div className={styles['dataset-card-date']}>
            <Calendar size={13}/>
            <span>{dataset.created_at ? new Date(dataset.created_at).toLocaleDateString() : dataset.lastModified}</span>
        </div>
    </div>
);

const DatasetDrawer = ({open, onClose}) => {
    const {datasets, loading, error, reload} = useDatasetContext();
    const [uploadOpen, setUploadOpen] = React.useState(false);
    const [editDataset, setEditDataset] = React.useState(null);

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
            PaperProps={{sx: {width: 340, boxShadow: 3}}}
        >
            <div className={styles['drawer-header']}>
                <div className={styles['drawer-title']}>
                    Data Management
                </div>
                <IconButton onClick={onClose} size="small">
                    <span style={{fontSize: 24, fontWeight: 700}}>&times;</span>
                </IconButton>
            </div>
            <Divider/>
            <div className={styles['drawer-content']}>
                <Button
                    variant="contained"
                    startIcon={<PlusCircle size={16}/>}
                    fullWidth
                    sx={{
                        mb: 2,
                        borderRadius: '12px',
                        fontWeight: 600,
                        textTransform: 'none'
                    }}
                    onClick={() => setUploadOpen(true)}
                >
                    Create Dataset
                </Button>
                <DatasetUploadModal isOpen={uploadOpen} onClose={() => setUploadOpen(false)}/>
                <DatasetEditModal open={!!editDataset} onClose={() => setEditDataset(null)} dataset={editDataset}
                                  onUpdated={reload}/>

                {loading && <Loading/>}
                {error && <ErrorMessage message={error.message}/>}

                <div className={styles['dataset-list']}>
                    {datasets.map(dataset => (
                        <DatasetCard
                            key={dataset.id}
                            dataset={dataset}
                            onDownload={handleDownload}
                            onDelete={handleDelete}
                            onEdit={setEditDataset}
                        />
                    ))}
                </div>

                {datasets.length === 0 && !loading && (
                    <EmptyState message="No datasets found."/>
                )}
            </div>
        </Drawer>
    );
};

export default DatasetDrawer;