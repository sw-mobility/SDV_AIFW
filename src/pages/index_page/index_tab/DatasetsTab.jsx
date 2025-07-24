import React, { useState, useEffect} from 'react';
import { Upload, Database, Tag, PlusCircle } from 'lucide-react';
import Card from '../../../components/common/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar, Download, Trash2 } from 'lucide-react';
import { fetchRawDatasets, fetchLabeledDatasets, downloadDataset, updateRawDataset, updateLabeledDataset, deleteDatasets, uploadRawFiles, uploadLabeledFiles } from '../../../api/datasets.js';
import Loading from '../../../components/common/Loading.jsx';
import ErrorMessage from '../../../components/common/ErrorMessage.jsx';
import ShowMoreGrid from '../../../components/common/ShowMoreGrid.jsx';
import DatasetUploadModal from '../../../components/dataset/DatasetUploadModal.jsx';
import DatasetDataPanel from '../../../components/dataset/DatasetDataPanel.jsx';
import { Edit2, Upload as UploadIcon } from 'lucide-react';
import Modal from '../../../components/common/Modal.jsx';
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
    // 데이터셋 상세/데이터 패널 상태
    const [dataPanelOpen, setDataPanelOpen] = useState(false);
    const [dataPanelTarget, setDataPanelTarget] = useState(null);
    // 카드 클릭 핸들러
    const handleCardClick = (dataset) => {
        setDataPanelTarget(dataset);
        setDataPanelOpen(true);
    };

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
                id: editData._id,
                uid: editData.uid || uid,
                name: fields.name,
                description: fields.description,
                type: fields.type,
                task_type: fields.taskType,
                label_format: fields.labelFormat
            });
            fetchLabeledDatasets({ uid }).then(res => setLabeledDatasets(res.data));
        } else {
            await updateRawDataset({
                id: editData._id || editData.id,
                uid: editData.uid || uid,
                name: fields.name,
                description: fields.description,
                type: fields.type
            });
            fetchRawDatasets({ uid }).then(res => setRawDatasets(res.data));
        }
        setEditOpen(false);
        setEditData(null);
    };

    const handleDelete = async (dataset) => {
        setDeletingId(dataset.did || dataset.id);
        try {
            const id = dataset._id;
            const path = dataset.file_path || dataset.path;
            await deleteDatasets({
                uid: uid,
                target_id_list: [id],
                target_path_list: path ? [path] : []
            });
            if (dataType === 'raw') {
                await fetchRawDatasets({ uid }).then(res => setRawDatasets(res.data));
            } else {
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
            await uploadLabeledFiles({ files, uid: uid, id: uploadTarget._id});
        } else {
            await uploadRawFiles({ files, uid: uid, id: uploadTarget._id});
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
        <Card className={styles.projectCard} onClick={() => handleCardClick({ ...dataset, _id: dataset._id || dataset.id, uid: uid, datasetType: isLabeled ? 'labeled' : 'raw' })}>
            <div className={styles.cardContent}>
                <div className={styles.cardIcon}>
                    {isLabeled ? <Tag size={18} color="var(--color-text-secondary)" /> : <Database size={18} color="var(--color-text-secondary)" />}
                </div>
                <div className={styles.cardName}>{dataset.name}</div>
                <div className={styles.cardDescription}>
                    {dataset.description ? dataset.description : <span style={{ color: '#bbb' }}>No description</span>}
                </div>
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
                    <button className={styles.actionButton} title="Edit" onClick={e => { e.stopPropagation(); handleEdit(dataset); }}>
                        <Edit2 size={16} />
                    </button>
                    <button className={styles.actionButton} title="Upload" onClick={e => { e.stopPropagation(); handleUpload(dataset); }}>
                        <UploadIcon size={14} />
                    </button>
                    <button className={styles.actionButton} title="Download" onClick={e => { e.stopPropagation(); handleDownload(dataset); }} disabled={downloadingId === dataset._id}>
                        {downloadingId === dataset._id ? <span>...</span> : <Download size={14} />}
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={e => { e.stopPropagation(); handleDelete(dataset); }} disabled={deletingId === dataset._id}>
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    // 정렬 함수: created_at, createdAt, created 중 하나라도 있으면 내림차순
    function sortByCreatedDesc(arr) {
        return [...arr].sort((a, b) => {
            const getTime = (d) => new Date(d.created_at || 0).getTime();
            return getTime(b) - getTime(a);
        });
    }

    const currentDatasets = dataType === 'raw' ? sortByCreatedDesc(rawDatasets) : sortByCreatedDesc(labeledDatasets);
    // 최신 데이터셋만 추출
    const latestDataset = currentDatasets[0];
    // 최신 데이터셋 카드
    const LatestDatasetCard = latestDataset ? (
        <DatasetCard key={latestDataset._id || latestDataset.id} dataset={latestDataset} isLabeled={dataType === 'labeled'} />
    ) : null;
    // 최신 데이터셋을 제외한 나머지 카드
    const restDatasetCards = latestDataset ? currentDatasets.slice(1).map(dataset => (
        <DatasetCard key={dataset._id || dataset.id} dataset={dataset} isLabeled={dataType === 'labeled'} />
    )) : [];
    // CreateDatasetCard 바로 오른쪽에 최신 데이터셋이 오도록
    const allDatasetCards = [
        <CreateDatasetCard key="create-dataset" />,
        ...(LatestDatasetCard ? [LatestDatasetCard] : []),
        ...restDatasetCards
    ];
    const visibleDatasetCards = showMore ? allDatasetCards : allDatasetCards.slice(0, cardsPerPage);

    if (loading) return <Loading />;
    if (error) return <ErrorMessage message={error} />;
    if (currentDatasets.length === 0) {
        return (
            <>
                <DatasetUploadModal isOpen={createOpen} onClose={() => setCreateOpen(false)} datasetType={dataType} onCreated={handleCreated} />
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
            <DatasetUploadModal isOpen={createOpen} onClose={() => setCreateOpen(false)} datasetType={dataType} onCreated={handleCreated} />
            {editOpen && (
                <DatasetUploadModal
                    isOpen={editOpen}
                    onClose={() => { setEditOpen(false); setEditData(null); }}
                    datasetType={dataType}
                    editMode
                    initialData={editData}
                    onSave={handleEditSave}
                    onCreated={handleCreated}
                />
            )}
            <UploadModal isOpen={uploadOpen} onClose={() => { setUploadOpen(false); setUploadTarget(null); }} onSave={handleUploadSave} />
            <DatasetDataPanel
                open={dataPanelOpen}
                onClose={() => setDataPanelOpen(false)}
                dataset={dataPanelTarget}
                datasetType={dataPanelTarget?.datasetType || dataType}
            />
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