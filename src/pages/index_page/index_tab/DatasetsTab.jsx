import React, { useState, useEffect, useRef } from 'react';
import { Upload, Database, Tag, PlusCircle } from 'lucide-react';
import Card from '../../../components/common/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar, Download, Trash2 } from 'lucide-react';
import { fetchRawDatasets, fetchLabeledDatasets, downloadDataset, createRawDataset, createLabeledDataset, updateRawDataset, updateLabeledDataset } from '../../../api/datasets.js';
import StatusChip from '../../../components/common/StatusChip.jsx';
import Loading from '../../../components/common/Loading.jsx';
import ErrorMessage from '../../../components/common/ErrorMessage.jsx';
import EmptyState from '../../../components/common/EmptyState.jsx';
import ShowMoreGrid from '../../../components/common/ShowMoreGrid.jsx';
import DatasetUploadModal from '../../../components/dataset/DatasetUploadModal.jsx';
import { Edit2, Upload as UploadIcon } from 'lucide-react';
import { deleteRawDatasets, deleteLabeledDatasets, uploadRawFiles, uploadLabeledFiles, getRawDataset, getLabeledDataset } from '../../../api/datasets.js';
import Modal from '../../../components/common/Modal.jsx';
import modalStyles from '../../../components/common/Modal.module.css';
import createModalStyles from '../../../components/common/CreateModal.module.css';
import { uid } from '../../../api/uid.js';
import FileUploadField from '../../../components/common/FileUploadField.jsx';

/**
 * DatasetsTab 컴포넌트
 *
 * "Raw" 또는 "Labeled" 데이터셋 목록을 탭 형식으로 표시하는 UI 컴포넌트
 *
 * - API로부터 데이터셋을 가져와 카드 형태로 렌더링
 * - 데이터 유형 전환(raw/labeled), 다운로드 및 삭제 기능 포함
 * - 로딩/에러/빈 상태 UI 처리
 *
 */


const CreateDatasetModal = ({ isOpen, onClose, datasetType, onCreated }) => {
    const [name, setName] = useState('');
    const [type, setType] = useState('Image');
    const [description, setDescription] = useState('');
    // Labeled 전용
    const [taskType, setTaskType] = useState('Classification');
    const [labelFormat, setLabelFormat] = useState('COCO');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);
    const resetForm = () => {
        setName('');
        setType('Image');
        setDescription('');
        setTaskType('Classification');
        setLabelFormat('COCO');
        setError(null);
        setSuccess(false);
    };
    useEffect(() => { if (isOpen) resetForm(); }, [isOpen]);
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setSuccess(false);
        try {
            if (datasetType === 'labeled') {
                await createLabeledDataset({
                    uid: uid,
                    name,
                    description,
                    type,
                    task_type: taskType,
                    label_format: labelFormat
                });
            } else {
                await createRawDataset({
                    uid: uid,
                    name,
                    description,
                    type
                });
            }
            setSuccess(true);
            onCreated && onCreated();
            setTimeout(() => {
                setSuccess(false);
                onClose();
            }, 1000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    if (!isOpen) return null;
    return (
        <div className={styles['modal-backdrop']}>
            <div className={styles['modal']}>
                <form onSubmit={handleSubmit} className={styles.formGroup}>
                    <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 12 }}>
                        {datasetType === 'labeled' ? 'Create Labeled Dataset' : 'Create Raw Dataset'}
                    </div>
                    <label className={styles.label}>
                        Name
                        <input type="text" value={name} onChange={e => setName(e.target.value)} className={styles.input} />
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
                        <textarea value={description} onChange={e => setDescription(e.target.value)} className={styles.input} rows={3} style={{ resize: 'vertical' }} />
                    </label>
                    {datasetType === 'labeled' && (
                        <>
                            <label className={styles.label}>
                                Task Type
                                <select value={taskType} onChange={e => setTaskType(e.target.value)} className={styles.input}>
                                    {['Classification', 'Detection', 'Segmentation', 'OCR', 'Other'].map(t => (
                                        <option key={t} value={t}>{t}</option>
                                    ))}
                                </select>
                            </label>
                            <label className={styles.label}>
                                Label Format
                                <select value={labelFormat} onChange={e => setLabelFormat(e.target.value)} className={styles.input}>
                                    {['COCO', 'VOC', 'YOLO', 'Custom'].map(f => (
                                        <option key={f} value={f}>{f}</option>
                                    ))}
                                </select>
                            </label>
                        </>
                    )}
                    {error && <div className={styles.fileError}>{error}</div>}
                    {success && <div className={styles.successMessage}>Created!</div>}
                    <div className={styles.modalActions}>
                        <button type="button" onClick={onClose} className={styles.cancelButton} disabled={loading}>Cancel</button>
                        <button type="submit" className={styles.cancelButton} style={{ background: '#2563eb', color: '#fff' }} disabled={loading || !name}>Create</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

const UploadModal = ({ isOpen, onClose, onSave }) => {
    const [files, setFiles] = useState([]);
    const [fileError, setFileError] = useState(null);
    const [loading, setLoading] = useState(false);
    const handleSubmit = async e => {
        e.preventDefault();
        setLoading(true);
        await onSave(files);
        setLoading(false);
        setFiles([]);
        setFileError(null);
    };
    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Upload Files">
            <form onSubmit={handleSubmit} className={createModalStyles.formGroup} style={{margin:0}}>
                <label className={createModalStyles.label}>
                    <FileUploadField files={files} setFiles={setFiles} fileError={fileError} setFileError={setFileError} accept={'.jpg,.jpeg,.png,.gif'} multiple />
                </label>
                {fileError && <div className={createModalStyles.fileError}>{fileError}</div>}
                <div className={createModalStyles.modalActions}>
                    <button type="button" onClick={onClose} className={createModalStyles.cancelButton} disabled={loading}>Cancel</button>
                    <button type="submit" className={createModalStyles.submitButton} disabled={loading || files.length === 0 || fileError}>Upload</button>
                </div>
            </form>
        </Modal>
    );
};

const DatasetsTab = () => {
    const [showMore, setShowMore] = useState(false);
    const [dataType, setDataType] = useState('raw');
    const cardsPerPage = 8;

    const [rawDatasets, setRawDatasets] = useState([]);
    const [labeledDatasets, setLabeledDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [createOpen, setCreateOpen] = useState(false);
    const [editOpen, setEditOpen] = useState(false);
    const [editData, setEditData] = useState(null);
    const [downloadingId, setDownloadingId] = useState(null);
    const [deletingId, setDeletingId] = useState(null);
    const [uploadOpen, setUploadOpen] = useState(false);
    const [uploadTarget, setUploadTarget] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        if (dataType === 'raw') {
            fetchRawDatasets({ uid })
                .then(res => setRawDatasets(res.data))
                .catch(err => setError(err.message))
                .finally(() => setLoading(false));
        } else {
            fetchLabeledDatasets({ uid })
                .then(res => setLabeledDatasets(res.data))
                .catch(err => setError(err.message))
                .finally(() => setLoading(false));
        }
    }, [dataType]);

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const handleDownload = async (dataset) => {
        setDownloadingId(dataset.id);
        try {
            await downloadDataset(dataset.id, dataType);
        } catch (err) {
            alert('Download failed: ' + err.message);
        } finally {
            setDownloadingId(null);
        }
    };

    const handleCreated = () => {
        if (dataType === 'labeled') {
            fetchLabeledDatasets({ uid }).then(res => setLabeledDatasets(res.data));
        } else {
            fetchRawDatasets({ uid }).then(res => setRawDatasets(res.data));
        }
    };

    const handleEdit = (dataset) => {
        setEditData(dataset);
        setEditOpen(true);
    };
    const handleEditSave = async (fields) => {
        if (dataType === 'labeled') {
            await updateLabeledDataset({
                did: editData.did || editData.id,
                uid: uid,
                name: fields.name,
                description: fields.description,
                type: fields.type,
                task_type: fields.taskType,
                label_format: fields.labelFormat
            });
            fetchLabeledDatasets({ uid }).then(res => setLabeledDatasets(res.data));
        } else {
            await updateRawDataset({
                did: editData.did || editData.id,
                name: fields.name,
                description: fields.description,
                type: fields.type
            });
            fetchRawDatasets({ uid }).then(res => setRawDatasets(res.data));
        }
        setEditOpen(false);
        setEditData(null);
    };

    const handleDelete = async (dataset, isRaw) => {
        setDeletingId(dataset.did || dataset.id);
        try {
            if (isRaw) {
                await deleteRawDatasets({ uid: uid, target_did_list: [dataset.did || dataset.id] });
                await fetchRawDatasets({ uid }).then(res => setRawDatasets(res.data));
            } else {
                await deleteLabeledDatasets({ uid: uid, target_did_list: [dataset.did || dataset.id] });
                await fetchLabeledDatasets({ uid }).then(res => setLabeledDatasets(res.data));
            }
        } finally {
            setDeletingId(null);
        }
    };

    const handleUpload = (dataset) => {
        setUploadTarget(dataset);
        setUploadOpen(true);
    };
    const handleUploadSave = async (files) => {
        if (dataType === 'labeled') {
            await uploadLabeledFiles({ files, uid: uid, did: uploadTarget.did || uploadTarget.id, task_type: uploadTarget.task_type || uploadTarget.taskType, label_format: uploadTarget.label_format || uploadTarget.labelFormat });
        } else {
            await uploadRawFiles({ files, uid: uid, did: uploadTarget.did || uploadTarget.id });
        }
        setUploadOpen(false);
        setUploadTarget(null);
        handleCreated();
    };

    const CreateDatasetCard = () => (
        <Card className={styles.createCard} onClick={() => setCreateOpen(true)}>
            <div className={styles.createCardContent}>
                <PlusCircle size={32} className={styles.createCardIcon} />
                <div className={styles.createCardText}>
                    Create Dataset
                </div>
            </div>
        </Card>
    );
    // Unified DatasetCard for both raw and labeled
    const DatasetCard = ({ dataset, isLabeled }) => (
        <Card className={styles.projectCard}>
            <div className={styles.cardContent}>
                <div className={styles.cardIcon}>
                    {isLabeled ? <Tag size={18} color="var(--color-text-secondary)" /> : <Database size={18} color="var(--color-text-secondary)" />}
                </div>
                <div className={styles.cardName}>{dataset.name}</div>
                {dataset.description && <div className={styles.cardDescription}>{dataset.description}</div>}
                <div className={styles.cardType}>
                    {dataset.type}
                    {isLabeled && dataset.task_type && <> / {dataset.task_type}</>}
                    {isLabeled && dataset.label_format && <> / {dataset.label_format}</>}
                </div>
                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {dataset.created_at && new Date(dataset.created_at).toLocaleDateString()}
                </div>
                <div className={styles.cardActions}>
                    <button className={styles.actionButton} title="Edit" onClick={() => handleEdit(dataset)}>
                        <Edit2 size={16} />
                    </button>
                    <button className={styles.actionButton} title="Upload" onClick={() => handleUpload(dataset)}>
                        <UploadIcon size={14} />
                    </button>
                    <button className={styles.actionButton} title="Download" onClick={() => handleDownload(dataset)} disabled={downloadingId === dataset._id}>
                        {downloadingId === dataset._id ? <span>...</span> : <Download size={14} />}
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={() => handleDelete(dataset, !isLabeled)} disabled={deletingId === dataset._id}>
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const currentDatasets = dataType === 'raw' ? rawDatasets : labeledDatasets;
    const allDatasetCards = [
        <CreateDatasetCard key="create-dataset" />,
        ...currentDatasets.map(dataset => (
            <DatasetCard key={dataset._id || dataset.id} dataset={dataset} isLabeled={dataType === 'labeled'} />
        ))
    ];
    const visibleDatasetCards = showMore ? allDatasetCards : allDatasetCards.slice(0, cardsPerPage);

    if (loading) return <Loading />;
    if (error) return <ErrorMessage message={error} />;
    if (currentDatasets.length === 0) {
        return (
            <>
                <DatasetUploadModal isOpen={createOpen} onClose={() => setCreateOpen(false)} datasetType={dataType} />
                <div className={styles.dataTypeToggle} style={{ marginBottom: 24 }}>
                    <button
                        className={`${styles.dataTypeButton} ${dataType === 'raw' ? styles.activeDataType : ''}`}
                        onClick={() => setDataType('raw')}
                    >
                        <Database size={16} />
                        Raw Data
                    </button>
                    <button
                        className={`${styles.dataTypeButton} ${dataType === 'labeled' ? styles.activeDataType : ''}`}
                        onClick={() => setDataType('labeled')}
                    >
                        <Tag size={16} />
                        Labeled Data
                    </button>
                </div>
                <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={false} onToggleShowMore={() => {}}>
                    <CreateDatasetCard key="create-dataset" />
                </ShowMoreGrid>
            </>
        );
    }

    return (
        <>
            <DatasetUploadModal isOpen={createOpen} onClose={() => setCreateOpen(false)} datasetType={dataType} />
            {editOpen && (
                <DatasetUploadModal
                    isOpen={editOpen}
                    onClose={() => { setEditOpen(false); setEditData(null); }}
                    datasetType={dataType}
                    editMode
                    initialData={editData}
                    onSave={handleEditSave}
                />
            )}
            <UploadModal isOpen={uploadOpen} onClose={() => { setUploadOpen(false); setUploadTarget(null); }} onSave={handleUploadSave} />
            <div className={styles.dataTypeToggle} style={{ marginBottom: 24 }}>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'raw' ? styles.activeDataType : ''}`}
                    onClick={() => setDataType('raw')}
                >
                    <Database size={16} />
                    Raw Data
                </button>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'labeled' ? styles.activeDataType : ''}`}
                    onClick={() => setDataType('labeled')}
                >
                    <Tag size={16} />
                    Labeled Data
                </button>
            </div>
            <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                {allDatasetCards}
            </ShowMoreGrid>
        </>
    );
};

export default DatasetsTab; 