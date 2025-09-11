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
import Loading from '../../ui/atoms/Loading.jsx';
import ErrorMessage from '../../ui/atoms/ErrorMessage.jsx';
import EmptyState from '../../ui/atoms/EmptyState.jsx';
import styles from './Dataset.module.css';
import { useDatasets } from '../../../hooks/index.js';
import DatasetUploadModal from './DatasetUploadModal.jsx';
import DatasetEditModal from './DatasetEditModal.jsx';
import UploadFilesModal from './DatasetUploadFilesModal.jsx';
import DatasetDataPanel from './DatasetDataPanel.jsx';
import DeleteConfirmModal from '../../ui/modals/DeleteConfirmModal.jsx';

/**
 * quick dataset management 버튼을 눌렀을 때 등장하는 mini dataset management입니다.
 * index page 의 dataset management 기능을 ui 배치만 바꿔 그대로 제공합니다.
 *
 * @type {{Image: React.JSX.Element, Text: React.JSX.Element, Audio: React.JSX.Element, Video: React.JSX.Element, Raw: React.JSX.Element, Graph: React.JSX.Element, Label: React.JSX.Element}}
 */

const typeIconMap = {
    Image: <ImageIcon size={14}/>,
    Text: <FileText size={14}/>,
    Audio: <AudioLines size={14}/>,
    Video: <Video size={14}/>,
    Raw: <Database size={14} style={{"color": "#3498db"}}/>,
    Graph: <Tag size={14}/>,
    Label: <Tag size={14} style={{"color": "#cc3030"}}/>,
};

const DatasetCard = ({dataset, onDownload, onDelete, onEdit, onUpload, onClick}) => (
    <div className={styles['dataset-card']} onClick={() => onClick(dataset)}>
        <div className={styles['dataset-card-header']}>
            <div>
                {dataset.datasetType === 'raw' ? typeIconMap.Raw : typeIconMap.Label}
            </div>
            <div className={styles['dataset-card-actions']}>
                <IconButton size="small" title="Edit" onClick={(e) => { e.stopPropagation(); onEdit(dataset); }}>
                    <Edit2 size={16}/>
                </IconButton>
                <IconButton size="small" title="Upload" onClick={(e) => { e.stopPropagation(); onUpload(dataset); }}>
                    <Upload size={16}/>
                </IconButton>
                <IconButton size="small" title="Download" onClick={(e) => { e.stopPropagation(); onDownload(dataset); }}>
                    <Download size={16}/>
                </IconButton>
                <IconButton size="small" title="Delete" onClick={(e) => { e.stopPropagation(); onDelete(dataset); }}>
                    <Trash2 size={16} style={{color: '#dc3545'}}/>
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
    const {
        dataType,
        loading,
        error,
        isCreateModalOpen,
        isEditModalOpen,
        isUploadModalOpen,
        isDataPanelOpen,
        isDeleteConfirmOpen,
        editData,
        dataPanelTarget,
        deleteTarget,
        handleDownload,
        handleEdit,
        openDeleteConfirm,
        confirmDelete,
        handleUpload,
        handleCardClick,
        handleDataTypeChange,
        openCreateModal,
        closeCreateModal,
        openEditModal,
        closeEditModal,
        openUploadModal,
        closeUploadModal,
        closeDataPanel,
        getCurrentDatasets,
        handleCreated,
        setIsDeleteConfirmOpen
    } = useDatasets();

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
                {/* Data Type Toggle */}
                <div className={styles.drawerDataTypeToggle}>
                    <button
                        className={`${styles.drawerDataTypeButton} ${dataType === 'raw' ? styles.active : ''}`}
                        onClick={() => handleDataTypeChange('raw')}
                    >
                        <Database size={14} />
                        Raw Data
                    </button>
                    <button
                        className={`${styles.drawerDataTypeButton} ${dataType === 'labeled' ? styles.active : ''}`}
                        onClick={() => handleDataTypeChange('labeled')}
                    >
                        <Tag size={14} />
                        Labeled Data
                    </button>
                </div>

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
                    onClick={openCreateModal}
                >
                    Create Dataset
                </Button>
                <DatasetUploadModal 
                    isOpen={isCreateModalOpen} 
                    onClose={closeCreateModal} 
                    datasetType={dataType}
                    onCreated={handleCreated}
                />

                {loading && <Loading/>}
                {error && <ErrorMessage message={error.message}/>}

                <div className={styles['dataset-list']}>
                    {getCurrentDatasets().map(dataset => (
                        <DatasetCard
                            key={dataset._id || dataset.id || dataset.did || `dataset-${dataset.name}-${dataset.created_at}`}
                            dataset={{...dataset, datasetType: dataType}}
                            onDownload={handleDownload}
                            onDelete={openDeleteConfirm}
                            onEdit={openEditModal}
                            onUpload={openUploadModal}
                            onClick={handleCardClick}
                        />
                    ))}
                </div>

                {getCurrentDatasets().length === 0 && !loading && (
                    <EmptyState message="No datasets found."/>
                )}
            </div>
            <UploadFilesModal
                isOpen={isUploadModalOpen}
                onClose={closeUploadModal}
                onSave={handleUpload}
            />
            {/* 편집 모달 */}
            <DatasetEditModal
                open={isEditModalOpen}
                onClose={closeEditModal}
                dataset={editData}
                datasetType={editData?.datasetType || dataType}
                onUpdated={handleCreated}
            />
            <DatasetDataPanel
                open={isDataPanelOpen}
                onClose={closeDataPanel}
                dataset={dataPanelTarget}
            />
            <DeleteConfirmModal
                isOpen={isDeleteConfirmOpen}
                onClose={() => setIsDeleteConfirmOpen(false)}
                onConfirm={confirmDelete}
                title="Delete Dataset"
                message="Are you sure you want to delete this dataset?"
                itemName={deleteTarget?.name}
            />
        </Drawer>
    );
};

export default DatasetDrawer;