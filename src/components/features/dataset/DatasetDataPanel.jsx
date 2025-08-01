import React from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../../ui/atoms/Button.jsx';
import Loading from '../../ui/atoms/Loading.jsx';
import ErrorMessage from '../../ui/atoms/ErrorMessage.jsx';
import EmptyState from '../../ui/atoms/EmptyState.jsx';
import Table from '../../ui/atoms/Table.jsx';
import FileUploadField from '../../ui/modals/FileUploadField.jsx';
import { useDatasetData } from '../../../hooks/dataset/useDatasetData.js';
import { Trash2, Download, Database, Tag } from 'lucide-react';

const DatasetDataPanel = ({ open, onClose, dataset }) => {
    const {
        data,
        loading,
        error,
        selected,
        uploading,
        uploadError,
        uploadFiles,
        showDeleteConfirm,
        downloading,
        handleSelect,
        handleSelectAll,
        handleDelete,
        handleUpload,
        handleDownloadDataset,
        handleDownloadSelected,
        updateUploadFiles,
        toggleDeleteConfirm,
        isLabeled
    } = useDatasetData(dataset, open);

    const titleIcon = isLabeled ? <Tag size={20} /> : <Database size={20} />;

    // Table columns/data 변환
    const columns = [
        '', 'Name', 'Type', 'Format', 'Created'
    ];

    // 전체선택 체크박스 별도 렌더링
    const renderSelectAll = () => (
        <input type="checkbox" checked={data?.data_list && Array.isArray(data.data_list) && selected.length === data.data_list.length && data.data_list.length > 0} onChange={handleSelectAll} key="all" />
    );

    const tableData = React.useMemo(() => {
        if (!data?.data_list || !Array.isArray(data.data_list)) {
            return [];
        }
        
        return data.data_list
            .filter(row => row && typeof row === 'object' && row._id)
            .map(row => ({
                _id: row._id || '',
                cells: [
                    <input 
                        type="checkbox" 
                        checked={selected.includes(row._id)} 
                        onChange={e => { e.stopPropagation(); handleSelect(row); }} 
                        onClick={e => e.stopPropagation()} 
                        key={row._id} 
                    />,
                    row.name || 'N/A',
                    row.type || 'N/A',
                    row.file_format || 'N/A',
                    row.created_at ? new Date(row.created_at).toLocaleString() : 'N/A'
                ]
            }));
    }, [data?.data_list, selected, handleSelect]);

    // rowKey는 _id로 지정
    return (
        <Modal isOpen={open} onClose={onClose} title="Dataset Details" titleIcon={titleIcon} className={styles.wideModal}>
            {loading && <Loading />}
            {error && <ErrorMessage message={error} />}
            {data && typeof data === 'object' && data.name && data.data_list && Array.isArray(data.data_list) && (
                <>
                    <div className={styles.detailInfo} style={{ marginBottom: 20, borderRadius: 12, background: '#f8f9fb', padding: 16, boxShadow: '0 1px 4px #0001', display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <div style={{ fontWeight: 600, fontSize: 18 }}>{data.name}</div>
                        <div style={{ color: '#888', marginBottom: 4 }}>{data.description || <span style={{ color: '#bbb' }}>No description</span>}</div>
                        <div style={{ display: 'flex', gap: 16, fontSize: 14, alignItems: 'center' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Type:</b> {data.type || 'N/A'}</span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Total:</b> {data.total || 0}</span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Created:</b> {(() => {
                                if (data && data.created_at) {
                                    return new Date(data.created_at).toLocaleString();
                                } else if (dataset && dataset.created_at) {
                                    return new Date(dataset.created_at).toLocaleString();
                                } else {
                                    return 'N/A';
                                }
                            })()}</span>
                        </div>
                    </div>
                    <form onSubmit={handleUpload} style={{ marginBottom: 16, display: 'flex', alignItems: 'flex-end', gap: 12, justifyContent: 'flex-end' }}>
                        <FileUploadField
                            files={uploadFiles}
                            setFiles={updateUploadFiles}
                            fileError={uploadError}
                            setFileError={() => {}} // useDatasetData에서 처리
                            accept="*"
                            multiple={true}
                        />
                        <Button
                            type="submit"
                            size="medium"
                            variant="primary"
                            disabled={uploading || !uploadFiles.length}
                            style={{ minWidth: 90 }}
                        >
                            {uploading ? <span style={{ display: 'inline-block', width: 16, height: 16, border: '2px solid #fff', borderTop: '2px solid #bfc6d1', borderRadius: '50%', animation: 'spin 1s linear infinite', verticalAlign: 'middle' }} /> : 'Upload'}
                        </Button>
                        <style>{`
                        @keyframes spin {
                          0% { transform: rotate(0deg); }
                          100% { transform: rotate(360deg); }
                        }
                        `}</style>
                    </form>
                    <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
                        <Button
                            size="medium"
                            variant="danger"
                            disabled={!selected.length || downloading}
                            onClick={toggleDeleteConfirm}
                        >
                            Delete Selected{selected.length > 0 ? ` (${selected.length})` : ''}
                        </Button>
                        <Button
                            size="medium"
                            variant="secondary"
                            disabled={!selected.length || downloading}
                            onClick={handleDownloadSelected}
                        >
                            <Download size={16} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                            {downloading ? 'Downloading...' : 'Download Selected'}
                        </Button>
                    </div>
                    <div style={{ overflowX: 'auto', width: '100%' }}>
                        <Table
                            columns={columns}
                            data={tableData.map(row => row.cells)}
                            rowKey="_id"
                            onRowClick={(_, idx) => {
                                if (!data?.data_list || !Array.isArray(data.data_list) || !data.data_list[idx]) return;
                                handleSelect(data.data_list[idx]);
                            }}
                            selectedId={null}
                            selectedRowClassName={styles.selectedRow}
                        />
                    </div>
                    {showDeleteConfirm && (
                        <Modal isOpen={showDeleteConfirm} onClose={toggleDeleteConfirm} title="Delete Data" className={styles.confirmModal}>
                            <div style={{ padding: 16, fontSize: 16, color: '#d32f2f', textAlign: 'center' }}>
                                <Trash2 size={32} style={{ marginBottom: 8 }} />
                                <div>Are you sure you want to delete the selected data?</div>
                                <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>This action cannot be undone.</div>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 16 }}>
                                <Button variant="secondary" onClick={toggleDeleteConfirm} size="medium">Cancel</Button>
                                <Button variant="danger" onClick={handleDelete} size="medium">Delete</Button>
                            </div>
                        </Modal>
                    )}
                </>
            )}
        </Modal>
    );
};

export default DatasetDataPanel; 