import React, { useState } from 'react';
import { Database, Tag, PlusCircle } from 'lucide-react';
import Card from '../../../components/ui/atoms/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar, Download, Trash2 } from 'lucide-react';
import Loading from '../../../components/ui/atoms/Loading.jsx';
import ErrorMessage from '../../../components/ui/atoms/ErrorMessage.jsx';
import EmptyState from '../../../components/ui/atoms/EmptyState.jsx';
import ShowMoreGrid from '../../../components/ui/atoms/ShowMoreGrid.jsx';
import DatasetUploadModal from '../../../components/features/dataset/DatasetUploadModal.jsx';
import DatasetEditModal from '../../../components/features/dataset/DatasetEditModal.jsx';
import DatasetDataPanel from '../../../components/features/dataset/DatasetDataPanel.jsx';
import DatasetUploadFilesModal from '../../../components/features/dataset/DatasetUploadFilesModal.jsx';
import DeleteConfirmModal from '../../../components/ui/modals/DeleteConfirmModal.jsx';
import { Edit2, Upload as UploadIcon } from 'lucide-react';
import { useDatasets } from '../../../hooks';

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




const DatasetsTab = () => {
    const [showMore, setShowMore] = useState(false);
    const cardsPerPage = 8;

    const {
        dataType,
        error,
        initialLoading,
        isCreateModalOpen,
        isEditModalOpen,
        isUploadModalOpen,
        isDataPanelOpen,
        isDeleteConfirmOpen,
        editData,
        dataPanelTarget,
        deleteTarget,
        downloadingId,
        deletingId,
        handleDownload,
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
        setIsDeleteConfirmOpen,
        getCurrentDatasets,
        handleCreated,
        uploadProgress
    } = useDatasets();

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const CreateDatasetCard = () => (
        <Card className={styles.createCard} onClick={openCreateModal}>
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
        <Card className={styles.projectCard} onClick={() => handleCardClick(dataset)}>
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
                    <button className={styles.actionButton} title="Edit" onClick={e => { e.stopPropagation(); openEditModal(dataset); }}>
                        <Edit2 size={16} />
                    </button>
                    <button className={styles.actionButton} title="Upload" onClick={e => { e.stopPropagation(); openUploadModal(dataset); }}>
                        <UploadIcon size={14} />
                    </button>
                    <button className={styles.actionButton} title="Download" onClick={e => { e.stopPropagation(); handleDownload(dataset); }} disabled={downloadingId === dataset._id}>
                        {downloadingId === dataset._id ? <span>...</span> : <Download size={14} />}
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={e => { e.stopPropagation(); openDeleteConfirm(dataset); }} disabled={deletingId === dataset._id}>
                        <Trash2 size={14} style={{color: '#dc3545'}} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const currentDatasets = getCurrentDatasets();
    console.log('Current datasets:', currentDatasets);
    console.log('Dataset details:', currentDatasets.map(d => ({ 
        name: d.name, 
        _id: d._id, 
        id: d.id, 
        did: d.did,
        type: d.type,
        created_at: d.created_at,
        key: d._id || d.id || d.did || `dataset-${d.name}-${d.created_at}`,
        fullObject: d
    })));
    
    // 최신 데이터셋만 추출
    const latestDataset = currentDatasets[0];
    // 최신 데이터셋 카드
    const LatestDatasetCard = latestDataset ? (
        <DatasetCard key={latestDataset._id || latestDataset.id || latestDataset.did || `dataset-${latestDataset.name}-${latestDataset.created_at}`} dataset={latestDataset} isLabeled={dataType === 'labeled'} />
    ) : null;
    // 최신 데이터셋을 제외한 나머지 카드
    const restDatasetCards = latestDataset ? currentDatasets.slice(1).map(dataset => (
        <DatasetCard key={dataset._id || dataset.id || dataset.did || `dataset-${dataset.name}-${dataset.created_at}`} dataset={dataset} isLabeled={dataType === 'labeled'} />
    )) : currentDatasets.map(dataset => (
        <DatasetCard key={dataset._id || dataset.id || dataset.did || `dataset-${dataset.name}-${dataset.created_at}`} dataset={dataset} isLabeled={dataType === 'labeled'} />
    ));
    // CreateDatasetCard 바로 오른쪽에 최신 데이터셋이 오도록
    const allDatasetCards = [
        <CreateDatasetCard key="create-dataset" />,
        ...(latestDataset ? [LatestDatasetCard] : []),
        ...restDatasetCards
    ];
    if (initialLoading) return <Loading fullHeight={true} />;
    if (error) return <ErrorMessage message={error} fullHeight={true} />;

    return (
        <>
            <DatasetUploadModal 
                key={`create-${dataType}-${isCreateModalOpen}`}
                isOpen={isCreateModalOpen} 
                onClose={closeCreateModal} 
                datasetType={dataType} 
                onCreated={handleCreated} 
            />
            {isEditModalOpen && (
                <DatasetEditModal
                    open={isEditModalOpen}
                    onClose={closeEditModal}
                    dataset={editData}
                    datasetType={dataType}
                    onUpdated={handleCreated}
                />
            )}
            <DatasetUploadFilesModal 
                isOpen={isUploadModalOpen} 
                onClose={closeUploadModal} 
                onSave={handleUpload}
                uploadProgress={uploadProgress}
            />
            <DatasetDataPanel
                open={isDataPanelOpen}
                onClose={closeDataPanel}
                dataset={dataPanelTarget}
                datasetType={dataPanelTarget?.datasetType || dataType}
            />
            <DeleteConfirmModal
                isOpen={isDeleteConfirmOpen}
                onClose={() => setIsDeleteConfirmOpen(false)}
                onConfirm={confirmDelete}
                title="Delete Dataset"
                message="Are you sure you want to delete this dataset?"
                itemName={deleteTarget?.name}
            />
            <div className={styles.dataTypeToggle} style={{ marginBottom: 24 }}>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'raw' ? styles.activeDataType : ''}`}
                    onClick={() => handleDataTypeChange('raw')}
                >
                    <Database size={16} />
                    Raw Data
                </button>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'labeled' ? styles.activeDataType : ''}`}
                    onClick={() => handleDataTypeChange('labeled')}
                >
                    <Tag size={16} />
                    Labeled Data
                </button>
            </div>
            {currentDatasets.length === 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '24px' }}>
                    <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                        {allDatasetCards}
                    </ShowMoreGrid>
                    <EmptyState message="No datasets found." fullHeight={false} />
                </div>
            ) : (
                <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                    {allDatasetCards}
                </ShowMoreGrid>
            )}
        </>
    );
};

export default DatasetsTab; 