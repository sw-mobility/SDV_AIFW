import { useState, useEffect } from 'react';
import { 
    fetchRawDatasets, 
    fetchLabeledDatasets, 
    downloadDatasetById, 
    updateRawDataset, 
    updateLabeledDataset, 
    deleteDatasets, 
    uploadRawFiles, 
    uploadLabeledFiles 
} from '../../api/datasets.js';
import { uid } from '../../api/uid.js';

/**
 * 데이터셋 관련 모든 로직을 관리하는 커스텀 훅
 * 
 * @returns {Object} 데이터셋 관련 상태와 핸들러
 */
export const useDatasets = () => {
    // 기본 상태
    const [dataType, setDataType] = useState('raw');
    const [rawDatasets, setRawDatasets] = useState([]);
    const [labeledDatasets, setLabeledDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [initialLoading, setInitialLoading] = useState(true);

    // 모달 상태
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [isDataPanelOpen, setIsDataPanelOpen] = useState(false);
    
    // 선택된 데이터
    const [editData, setEditData] = useState(null);
    const [uploadTarget, setUploadTarget] = useState(null);
    const [dataPanelTarget, setDataPanelTarget] = useState(null);

    // 액션 상태
    const [downloadingId, setDownloadingId] = useState(null);
    const [deletingId, setDeletingId] = useState(null);
    
    // Show More 상태
    const [showMoreCount, setShowMoreCount] = useState(5);

    // 데이터셋 목록 조회
    const fetchDatasetsList = async () => {
        setInitialLoading(true);
        setError(null);
        
        try {
            const [rawRes, labeledRes] = await Promise.all([
                fetchRawDatasets({ uid }),
                fetchLabeledDatasets({ uid })
            ]);
            
            setRawDatasets(rawRes.data || []);
            setLabeledDatasets(labeledRes.data || []);
        } catch (err) {
            console.error('DatasetsTab: Error fetching datasets:', err);
            setError(err.message || 'Failed to fetch datasets');
        } finally {
            setInitialLoading(false);
        }
    };

    // 초기 로드
    useEffect(() => {
        fetchDatasetsList();
    }, []);

    // 데이터 타입 변경 시 해당 데이터셋 목록 새로고침
    const refreshCurrentDatasets = async () => {
        setLoading(true);
        setError(null);
        
        try {
            if (dataType === 'raw') {
                const res = await fetchRawDatasets({ uid });
                setRawDatasets(res.data || []);
            } else {
                const res = await fetchLabeledDatasets({ uid });
                setLabeledDatasets(res.data || []);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 데이터셋 다운로드
    const handleDownload = async (dataset) => {
        setDownloadingId(dataset._id || dataset.id);
        try {
            await downloadDatasetById({ 
                uid: dataset.uid || uid, 
                target_id: dataset._id || dataset.id 
            });
        } catch (err) {
            alert('Download failed: ' + err.message);
        } finally {
            setDownloadingId(null);
        }
    };

    // 데이터셋 수정
    const handleEdit = async (fields) => {
        try {
            setLoading(true);
            setError(null);
            
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
            } else {
                await updateRawDataset({
                    id: editData._id || editData.id,
                    uid: editData.uid || uid,
                    name: fields.name,
                    description: fields.description,
                    type: fields.type
                });
            }
            
            await refreshCurrentDatasets();
            setIsEditModalOpen(false);
            setEditData(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 데이터셋 삭제
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
            
            await refreshCurrentDatasets();
        } catch (err) {
            setError(err.message);
        } finally {
            setDeletingId(null);
        }
    };

    // 파일 업로드
    const handleUpload = async (files) => {
        try {
            setLoading(true);
            setError(null);
            
            if (dataType === 'labeled') {
                await uploadLabeledFiles({ 
                    files, 
                    uid: uid, 
                    id: uploadTarget._id
                });
            } else {
                await uploadRawFiles({ 
                    files, 
                    uid: uid, 
                    id: uploadTarget._id
                });
            }
            
            setIsUploadModalOpen(false);
            setUploadTarget(null);
            await refreshCurrentDatasets();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 데이터셋 생성/수정 후 콜백
    const handleCreated = async () => {
        await refreshCurrentDatasets();
    };

    // 카드 클릭 (데이터 패널 열기)
    const handleCardClick = (dataset) => {
        setDataPanelTarget({ 
            ...dataset, 
            _id: dataset._id || dataset.id || dataset.did, 
            uid: uid, 
            datasetType: dataType 
        });
        setIsDataPanelOpen(true);
    };

    // 모달 핸들러
    const openCreateModal = () => setIsCreateModalOpen(true);
    const closeCreateModal = () => setIsCreateModalOpen(false);
    
    const openEditModal = (dataset) => {
        setEditData(dataset);
        setIsEditModalOpen(true);
    };
    
    const closeEditModal = () => {
        setIsEditModalOpen(false);
        setEditData(null);
    };
    
    const openUploadModal = (dataset) => {
        setUploadTarget(dataset);
        setIsUploadModalOpen(true);
    };
    
    const closeUploadModal = () => {
        setIsUploadModalOpen(false);
        setUploadTarget(null);
    };
    
    const openDataPanel = (dataset) => {
        setDataPanelTarget(dataset);
        setIsDataPanelOpen(true);
    };
    
    const closeDataPanel = () => {
        setIsDataPanelOpen(false);
        setDataPanelTarget(null);
    };

    // 데이터 타입 변경
    const handleDataTypeChange = (newDataType) => {
        setDataType(newDataType);
        setShowMoreCount(5); // 데이터 타입 변경 시 Show More 카운트 리셋
    };

    // 현재 데이터셋 목록 (정렬된)
    const getCurrentDatasets = () => {
        const datasets = dataType === 'raw' ? rawDatasets : labeledDatasets;
        return [...datasets].sort((a, b) => {
            const getTime = (d) => new Date(d.created_at || 0).getTime();
            return getTime(b) - getTime(a);
        });
    };
    
    // Show More를 위한 데이터셋 목록 (제한된 개수)
    const getLimitedDatasets = () => {
        return getCurrentDatasets().slice(0, showMoreCount);
    };
    
    // Show More 핸들러
    const handleShowMore = () => {
        setShowMoreCount(prev => prev + 5);
    };
    
    // Show More 버튼 표시 여부
    const shouldShowMoreButton = () => {
        return getCurrentDatasets().length > showMoreCount;
    };

    return {
        // 상태
        dataType,
        rawDatasets,
        labeledDatasets,
        loading,
        error,
        initialLoading,
        isCreateModalOpen,
        isEditModalOpen,
        isUploadModalOpen,
        isDataPanelOpen,
        editData,
        uploadTarget,
        dataPanelTarget,
        downloadingId,
        deletingId,
        
        // 핸들러
        handleDownload,
        handleEdit,
        handleDelete,
        handleUpload,
        handleCardClick,
        handleDataTypeChange,
        handleCreated,
        
        // 모달 핸들러
        openCreateModal,
        closeCreateModal,
        openEditModal,
        closeEditModal,
        openUploadModal,
        closeUploadModal,
        openDataPanel,
        closeDataPanel,
        
        // 유틸리티
        fetchDatasetsList,
        refreshCurrentDatasets,
        getCurrentDatasets,
        getLimitedDatasets,
        handleShowMore,
        shouldShowMoreButton
    };
}; 