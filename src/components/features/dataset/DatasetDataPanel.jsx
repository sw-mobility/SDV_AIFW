import React, { useEffect, useState } from 'react';
import Modal from '../../ui/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../../ui/Button.jsx';
import Loading from '../../ui/Loading.jsx';
import ErrorMessage from '../../ui/ErrorMessage.jsx';
import EmptyState from '../../ui/EmptyState.jsx';
import Table from '../../ui/Table.jsx';
import FileUploadField from '../../ui/FileUploadField.jsx';
import {getRawDataset, uploadRawFiles, getLabeledDataset, uploadLabeledFiles, deleteData, downloadDatasetById, downloadDataByPaths} from '../../../api/datasets.js';
import { Trash2, Download } from 'lucide-react';

const DatasetDataPanel = ({ open, onClose, dataset }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selected, setSelected] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [uploadFiles, setUploadFiles] = useState([]);
    const [refreshKey, setRefreshKey] = useState(0);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [downloading, setDownloading] = useState(false);

    useEffect(() => {
        if (!open || !dataset) return;
        setLoading(true);
        setError(null);
        const isLabeled = dataset.datasetType === 'labeled' || dataset.type === 'labeled';
        const fetchData = isLabeled
          ? getLabeledDataset({ did: dataset.did || dataset._id || dataset.id, id: dataset._id || dataset.id, uid: dataset.uid || '' })
          : getRawDataset({ did: dataset.did || dataset._id || dataset.id, id: dataset._id || dataset.id, uid: dataset.uid || '' });
        fetchData
            .then(res => setData(res))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    }, [open, dataset, refreshKey]);

    // 체크박스 핸들러
    const handleSelect = (row) => {
        setSelected(prev => prev.includes(row._id) ? prev.filter(x => x !== row._id) : [...prev, row._id]);
    };
    const handleSelectAll = () => {
        if (!data?.data_list) return;
        if (selected.length === data.data_list.length) setSelected([]);
        else setSelected(data.data_list.map(d => d._id));
    };
    const isLabeled = dataset && (dataset.datasetType === 'labeled' || dataset.type === 'labeled');
    const handleDelete = async () => {
        if (!selected.length) return;
        setShowDeleteConfirm(false);
        await deleteData({ uid: dataset.uid || '', id: dataset._id , target_id_list: selected });
        setSelected([]);
        setRefreshKey(k => k + 1);
    };
    const handleUpload = async (e) => {
        e.preventDefault();
        if (!uploadFiles.length) return;
        setUploading(true);
        setUploadError(null);
        try {
            if (isLabeled) {
                await uploadLabeledFiles({ files: uploadFiles, uid: dataset.uid || '', id: dataset._id, task_type: dataset.task_type || dataset.taskType, label_format: dataset.label_format || dataset.labelFormat });
            } else {
            await uploadRawFiles({ files: uploadFiles, uid: dataset.uid || '', id:dataset._id});
            }
            setUploadFiles([]);
            setRefreshKey(k => k + 1);
        } catch (err) {
            setUploadError(err.message);
        } finally {
            setUploading(false);
        }
    };
    const handleDownloadDataset = async () => {
        if (!dataset?._id || !dataset?.uid) return;
        setDownloading(true);
        try {
            await downloadDatasetById({ uid: dataset.uid, target_id: dataset._id });
        } catch (e) {
            alert('Download failed: ' + e.message);
        } finally {
            setDownloading(false);
        }
    };
    const handleDownloadSelected = async () => {
        if (!selected.length || !data?.data_list) return;
        setDownloading(true);
        try {
            const selectedPaths = data.data_list.filter(d => selected.includes(d._id) && d.path).map(d => d.path);
            if (selectedPaths.length === 0) throw new Error('No valid data paths');
            await downloadDataByPaths({ uid: dataset.uid, target_path_list: selectedPaths });
        } catch (e) {
            alert('Download failed: ' + e.message);
        } finally {
            setDownloading(false);
        }
    };

    // Table columns/data 변환
    const columns = [
        '', 'Name', 'Type', 'Format', 'Created'
    ];
    // 전체선택 체크박스 별도 렌더링
    const renderSelectAll = () => (
        <input type="checkbox" checked={data?.data_list && selected.length === data.data_list.length && data.data_list.length > 0} onChange={handleSelectAll} key="all" />
    );
    const tableData = (data?.data_list || []).map(row => ({
        _id: row._id,
        cells: [
            <input type="checkbox" checked={selected.includes(row._id)} onChange={e => { e.stopPropagation(); handleSelect(row); }} onClick={e => e.stopPropagation()} key={row._id} />,
            row.name,
            row.type,
            row.file_format,
            row.created_at && new Date(row.created_at).toLocaleString()
        ]
    }));

    // rowKey는 _id로 지정
    return (
        <Modal isOpen={open} onClose={onClose} title={data ? data.name : 'Dataset Details'} className={styles.wideModal}>
            {loading && <Loading />}
            {error && <ErrorMessage message={error} />}
            {data && (
                <>
                    <div className={styles.detailInfo} style={{ marginBottom: 20, borderRadius: 12, background: '#f8f9fb', padding: 16, boxShadow: '0 1px 4px #0001', display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <div style={{ fontWeight: 600, fontSize: 18 }}>{data.name}</div>
                        <div style={{ color: '#888', marginBottom: 4 }}>{data.description || <span style={{ color: '#bbb' }}>No description</span>}</div>
                        <div style={{ display: 'flex', gap: 16, fontSize: 14 }}>
                            <span><b>Type:</b> {data.type}</span>
                            <span><b>Total:</b> {data.total}</span>
                            <span><b>Created:</b> {data.created_at && new Date(data.created_at).toLocaleString()}</span>
                        </div>
                    </div>
                    <form onSubmit={handleUpload} style={{ marginBottom: 16, display: 'flex', alignItems: 'flex-end', gap: 12, justifyContent: 'flex-end' }}>
                        <FileUploadField
                            files={uploadFiles}
                            setFiles={setUploadFiles}
                            fileError={uploadError}
                            setFileError={setUploadError}
                            accept={'.jpg,.jpeg,.png,.gif'}
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
                            onClick={() => setShowDeleteConfirm(true)}
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
                            if (!data.data_list || !data.data_list[idx]) return;
                            handleSelect(data.data_list[idx]);
                        }}
                        selectedId={null}
                        selectedRowClassName={styles.selectedRow}
                    />
                    </div>
                    {(data.data_list && data.data_list.length === 0) && <EmptyState message="No data in this dataset." />}
                    {showDeleteConfirm && (
                        <Modal isOpen={showDeleteConfirm} onClose={() => setShowDeleteConfirm(false)} title="Delete Data" className={styles.confirmModal}>
                            <div style={{ padding: 16, fontSize: 16, color: '#d32f2f', textAlign: 'center' }}>
                                <Trash2 size={32} style={{ marginBottom: 8 }} />
                                <div>Are you sure you want to delete the selected data?</div>
                                <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>This action cannot be undone.</div>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 16 }}>
                                <Button variant="secondary" onClick={() => setShowDeleteConfirm(false)} size="medium">Cancel</Button>
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