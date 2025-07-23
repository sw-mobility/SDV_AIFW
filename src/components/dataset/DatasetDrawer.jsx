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
    Video,
    Upload
} from 'lucide-react';
import {useDatasetContext} from '../../context/DatasetContext.jsx';
import Loading from '../common/Loading.jsx';
import ErrorMessage from '../common/ErrorMessage.jsx';
import EmptyState from '../common/EmptyState.jsx';
import styles from './Dataset.module.css';
import {downloadDataset, updateLabeledDataset, updateRawDataset, uploadRawFiles, deleteDatasets} from '../../api/datasets.js';
import { uid } from '../../api/uid.js';
import DatasetUploadModal from './DatasetUploadModal.jsx';
import UploadFilesModal from './DatasetUploadFilesModal.jsx';
import DatasetDataPanel from './DatasetDataPanel.jsx';

const typeIconMap = {
    Image: <ImageIcon size={14}/>,
    Text: <FileText size={14}/>,
    Audio: <AudioLines size={14}/>,
    Video: <Video size={14}/>,
    Raw: <Database size={14} style={{"color": "#3498db"}}/>,
    Graph: <Tag size={14}/>,
    Label: <Tag size={14} style={{"color": "#cc305a"}}/>,
};

const DatasetCard = ({dataset, onDownload, onDelete, onEdit, onUpload, onClick}) => (
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
                {dataset.datasetType === 'raw' && (
                    <IconButton size="small" title="Upload" onClick={() => onUpload(dataset)}>
                        <Upload size={16}/>
                    </IconButton>
                )}
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
    const [uploadTarget, setUploadTarget] = React.useState(null);
    const [createOpen, setCreateOpen] = React.useState(false);
    const [editOpen, setEditOpen] = React.useState(false);
    const [editTarget, setEditTarget] = React.useState(null);
    // 데이터셋 상세/데이터 패널 상태
    const [dataPanelOpen, setDataPanelOpen] = React.useState(false);
    const [dataPanelTarget, setDataPanelTarget] = React.useState(null);
    // 카드 클릭 핸들러
    const handleCardClick = (dataset) => {
        setDataPanelTarget(dataset);
        setDataPanelOpen(true);
    };

    const handleDownload = async (dataset) => {
        try {
            await downloadDataset(dataset.id, dataset.type === 'Image' ? 'raw' : 'labeled');
            console.log('Download started for:', dataset.name);
        } catch (error) {
            console.error('Download failed:', error.message);
        }
    };

    const handleDelete = async (dataset) => {
        try {
            const id = dataset._id;
            const path = dataset.file_path || dataset.path;
            await deleteDatasets({
                uid: uid,
                target_id_list: [id],
                target_path_list: path ? [path] : []
            });
            reload();
        } catch (error) {
            console.error('Delete failed:', error.message);
        }
    };

    // 업로드 핸들러 (파일 업로드용)
    const handleUpload = (dataset) => {
        setUploadTarget(dataset);
        setUploadOpen(true);
    };
    // 업로드 저장 핸들러 (실제 API 호출)
    const handleUploadSave = async (files) => {
        if (uploadTarget?.datasetType === 'labeled') {
            await uploadLabeledFiles({ files, uid: uploadTarget.uid || '', id: uploadTarget._id, task_type: uploadTarget.task_type, label_format: uploadTarget.label_format });
        } else {
            await uploadRawFiles({ files, uid: uploadTarget.uid || '', did: uploadTarget.did || uploadTarget.id });
        }
        setUploadOpen(false);
        setUploadTarget(null);
        reload();
    };

    // edit 버튼 클릭 시
    const handleEdit = (dataset) => {
        setEditTarget(dataset);
        setEditOpen(true);
    };

    const handleEditSave = async (fields) => {
        if (editTarget?.datasetType === 'labeled') {
            await updateLabeledDataset({
                id: editTarget._id,
                uid: uid,
                name: fields.name,
                description: fields.description,
                type: fields.type,
                task_type: fields.taskType,
                label_format: fields.labelFormat
            });
        } else {
            await updateRawDataset({
                id:editTarget._id,
                name: fields.name,
                description: fields.description,
                type: fields.type
            });
        }
        setEditOpen(false);
        setEditTarget(null);
        reload();
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
                    onClick={() => setCreateOpen(true)}
                >
                    Create Dataset
                </Button>
                <DatasetUploadModal isOpen={createOpen} onClose={() => setCreateOpen(false)} />

                {loading && <Loading/>}
                {error && <ErrorMessage message={error.message}/>}

                <div className={styles['dataset-list']}>
                    {datasets.map(dataset => (
                        <DatasetCard
                            key={dataset.id}
                            dataset={dataset}
                            onDownload={handleDownload}
                            onDelete={handleDelete}
                            onEdit={handleEdit}
                            onUpload={handleUpload}
                            onClick={handleCardClick}
                        />
                    ))}
                </div>

                {datasets.length === 0 && !loading && (
                    <EmptyState message="No datasets found."/>
                )}
            </div>
            <UploadFilesModal
                isOpen={uploadOpen}
                onClose={() => { setUploadOpen(false); setUploadTarget(null); }}
                onSave={handleUploadSave}
            />
            {/* edit 모달도 DatasetUploadModal로 통일 */}
            <DatasetUploadModal
                isOpen={editOpen}
                onClose={() => setEditOpen(false)}
                editMode={true}
                datasetType={editTarget?.datasetType || 'raw'}
                initialData={editTarget || {}}
                onSave={handleEditSave}
                onCreated={reload}
            />
            <DatasetDataPanel
                open={dataPanelOpen}
                onClose={() => setDataPanelOpen(false)}
                dataset={dataPanelTarget}
            />
        </Drawer>
    );
};

export default DatasetDrawer;