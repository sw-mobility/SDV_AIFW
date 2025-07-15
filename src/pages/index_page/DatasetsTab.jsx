import React, { useState, useEffect } from 'react';
import { Upload, Database, Tag} from 'lucide-react';
import Card from '../../components/common/Card.jsx';
import styles from './IndexPage.module.css';
import { Calendar, Download, Trash2 } from 'lucide-react';
import { fetchRawDatasets, fetchLabeledDatasets, downloadDataset } from '../../api/datasets.js';
import StatusChip from '../../components/common/StatusChip.jsx';
import Loading from '../../components/common/Loading.jsx';
import ErrorMessage from '../../components/common/ErrorMessage.jsx';
import EmptyState from '../../components/common/EmptyState.jsx';
import ShowMoreGrid from '../../components/common/ShowMoreGrid.jsx';
import DatasetUploadModal from '../../components/dataset/DatasetUploadModal.jsx';

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


const DatasetsTab = ({ mockState }) => {
    const [showMore, setShowMore] = useState(false);
    const [dataType, setDataType] = useState('raw');
    const cardsPerPage = 8;

    const [rawDatasets, setRawDatasets] = useState([]);
    const [labeledDatasets, setLabeledDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [uploadOpen, setUploadOpen] = useState(false);
    const [downloadingId, setDownloadingId] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        if (dataType === 'raw') { //dataType 바뀔 때마다 fetch API 실행
            fetchRawDatasets(mockState)
                .then(res => setRawDatasets(res.data))
                .catch(err => setError(err.message))
                .finally(() => setLoading(false));
        } else {
            fetchLabeledDatasets(mockState)
                .then(res => setLabeledDatasets(res.data))
                .catch(err => setError(err.message))
                .finally(() => setLoading(false));
        }
    }, [dataType, mockState]);

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

    const handleDelete = (dataset, isRaw) => {
        if (isRaw) {
            setRawDatasets(prev => prev.filter(d => d.id !== dataset.id));
        } else {
            setLabeledDatasets(prev => prev.filter(d => d.id !== dataset.id));
        }
    };

    const CreateDatasetCard = () => (
        <Card className={styles.createCard} onClick={() => setUploadOpen(true)}>
            <div className={styles.createCardContent}>
                <Upload size={32} className={styles.createCardIcon} />
                <div className={styles.createCardText}>
                    Upload New Dataset
                </div>
            </div>
        </Card>
    );

    const RawDatasetCard = ({ dataset }) => (
        <Card className={styles.datasetCard}>
            <div className={styles.cardContent}>
                <StatusChip status={dataset.status} className={styles.statusChip} />

                <div className={styles.cardIcon}>
                    <Database size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {dataset.name}
                </div>

                <div className={styles.cardInfo}>
                    <span className={styles.cardType}>{dataset.type}</span>
                    <span className={styles.cardSize}>{dataset.size}</span>
                </div>

                <div style={{ height: '24px' }}></div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {dataset.lastModified}
                </div>

                <div className={styles.cardActions}>
                    <button className={styles.actionButton} title="Download" onClick={() => handleDownload(dataset)} disabled={downloadingId === dataset.id}>
                        {downloadingId === dataset.id ? <span>...</span> : <Download size={14} />}
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={() => handleDelete(dataset, true)}>
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const LabeledDatasetCard = ({ dataset }) => (
        <Card className={styles.datasetCard}>
            <div className={styles.cardContent}>
                <StatusChip status={dataset.status} className={styles.statusChip} />

                <div className={styles.cardIcon}>
                    <Tag size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {dataset.name}
                </div>

                <div className={styles.cardInfo}>
                    <span className={styles.cardType}>{dataset.type}</span>
                    <span className={styles.cardSize}>{dataset.size}</span>
                </div>

                <div className={styles.cardLabelCount}>
                    <Tag size={14} />
                    {dataset.labelCount.toLocaleString()} labels
                </div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {dataset.lastModified}
                </div>

                <div className={styles.cardActions}>
                    <button className={styles.actionButton} title="Download" onClick={() => handleDownload(dataset)} disabled={downloadingId === dataset.id}>
                        {downloadingId === dataset.id ? <span>...</span> : <Download size={14} />}
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={() => handleDelete(dataset, false)}>
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
            dataType === 'raw' ? 
            <RawDatasetCard key={dataset.id} dataset={dataset} /> :
            <LabeledDatasetCard key={dataset.id} dataset={dataset} />
        ))
    ];
    const visibleDatasetCards = showMore ? allDatasetCards : allDatasetCards.slice(0, cardsPerPage);

    if (loading || mockState?.loading) return <Loading />;
    if (error || mockState?.error) return <ErrorMessage message={error || 'Mock error!'} />;
    if (currentDatasets.length === 0 || mockState?.empty) return <EmptyState message="No datasets found." />;

    return (
        <>
            <DatasetUploadModal isOpen={uploadOpen} onClose={() => setUploadOpen(false)} />
            <div className={styles.dataTypeToggle}>
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