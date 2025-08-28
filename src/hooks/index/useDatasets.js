import { useState, useEffect} from 'react';
import { 
    fetchRawDatasets, 
    fetchLabeledDatasets, 
    downloadDatasetById, 
    updateRawDataset, 
    updateLabeledDataset, 
    deleteDatasets,
    uploadRawFilesInBatches,
    uploadLabeledFilesInBatches
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
    const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
    
    // 선택된 데이터
    const [editData, setEditData] = useState(null);
    const [uploadTarget, setUploadTarget] = useState(null);
    const [dataPanelTarget, setDataPanelTarget] = useState(null);
    const [deleteTarget, setDeleteTarget] = useState(null);

    // 액션 상태
    const [downloadingId, setDownloadingId] = useState(null);
    const [deletingId, setDeletingId] = useState(null);
    
    // 업로드 진행률 상태
    const [uploadProgress, setUploadProgress] = useState(null);

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
                target_did: dataset._id || dataset.id || dataset.did,
                dataset_name: dataset.name
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
            
            console.log('Edit data:', editData);
            console.log('Fields:', fields);
            console.log('Data type:', dataType);
            
            if (dataType === 'labeled') {
                const updateData = {
                    did: editData._id,
                    uid: editData.uid || uid,
                    name: fields.name,
                    description: fields.description,
                    type: fields.type,
                    task_type: fields.taskType,
                    label_format: fields.labelFormat
                };
                console.log('Update labeled dataset with:', updateData);
                await updateLabeledDataset(updateData);
            } else {
                const updateData = {
                    did: editData._id || editData.id,
                    uid: editData.uid || uid,
                    name: fields.name,
                    description: fields.description,
                    type: fields.type
                };
                console.log('Update raw dataset with:', updateData);
                await updateRawDataset(updateData);
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

    // 데이터셋 삭제 확인 모달 열기
    const openDeleteConfirm = (dataset) => {
        setDeleteTarget(dataset);
        setIsDeleteConfirmOpen(true);
    };

    // 데이터셋 삭제 실행
    const handleDelete = async (dataset) => {
        const datasetId = dataset._id || dataset.id || dataset.did;
        setDeletingId(datasetId);
        setError(null); // 이전 에러 초기화
        
        try {
            console.log('Full dataset object:', dataset);
            console.log('Available dataset fields:', Object.keys(dataset));
            console.log('Deleting dataset:', { 
                id: datasetId, 
                path: dataset.file_path, 
                dataset,
                uid: uid
            });
            
            // Check for various possible path fields
            const possiblePathFields = ['file_path', 'path', 'filePath', 'data_path', 'dataPath', 'storage_path', 'storagePath'];
            let targetPath = null;
            
            for (const field of possiblePathFields) {
                if (dataset[field]) {
                    targetPath = dataset[field];
                    console.log(`Found path in field '${field}':`, targetPath);
                    break;
                }
            }
            
            if (!targetPath) {
                console.log('No path field found, using dataset ID as path');
                targetPath = datasetId;
            }
            
            // 삭제 요청 전송
            const deleteRequest = {
                uid: uid,
                target_did_list: [datasetId],
                target_path_list: [targetPath]
            };
            
            console.log('Delete request payload:', deleteRequest);
            
            await deleteDatasets(deleteRequest);
            
            console.log('Dataset deleted successfully');
            
            // 삭제 완료 후 데이터 새로고침
            await fetchDatasetsList();
        } catch (err) {
            console.error('Delete error:', err);
            
            // "삭제된 문서가 없습니다" 메시지 처리
            if (err.message.includes('삭제된 문서가 없습니다') || err.message.includes('No documents found to delete')) {
                console.log('No documents to delete - treating as success');
                // 에러를 표시하지 않고 성공으로 처리
                await fetchDatasetsList();
                return;
            }
            
            // 404 에러나 resources not found 메시지 처리
            if (err.message.includes('resources not found') || err.message.includes('404')) {
                console.log('Resources not found - treating as success');
                // 에러를 표시하지 않고 성공으로 처리
                await fetchDatasetsList();
                return;
            }
            
            // AWS S3 에러인 경우 사용자 친화적인 메시지 표시
            if (err.message.includes('MalformedXML') || err.message.includes('DeleteObjects')) {
                setError('Failed to delete dataset files. This might be a temporary issue. Please try again in a moment.');
            } else {
                setError(err.message || 'Failed to delete dataset. Please try again.');
            }
        } finally {
            setDeletingId(null);
        }
    };

    // 삭제 확인 모달에서 삭제 실행
    const confirmDelete = async () => {
        if (deleteTarget) {
            await handleDelete(deleteTarget);
            setIsDeleteConfirmOpen(false);
            setDeleteTarget(null);
        }
    };

    // 파일 업로드
    const handleUpload = async (files) => {
        try {
            setLoading(true);
            setError(null);
            setUploadProgress(null);
            
            // 모든 업로드에 배치 업로드 사용 (진행률 표시를 위해)
            const batchSize = files.length > 1000 ? 1000 : Math.max(1, Math.ceil(files.length / 5)); // 최소 1개, 최대 5개 배치
            
            if (dataType === 'labeled') {
                await uploadLabeledFilesInBatches({ 
                    files, 
                    uid: uid, 
                    did: uploadTarget._id,
                    batchSize,
                    onProgress: (progress) => {
                        // 진행률 업데이트
                        setUploadProgress(progress);
                    }
                });
            } else {
                await uploadRawFilesInBatches({ 
                    files, 
                    uid: uid, 
                    did: uploadTarget._id,
                    batchSize,
                    onProgress: (progress) => {
                        // 진행률 업데이트
                        setUploadProgress(progress);
                    }
                });
            }
            
            // 업로드 완료 후 서버 처리 시간을 위한 지연
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            setIsUploadModalOpen(false);
            setUploadTarget(null);
            setUploadProgress(null);
            
            // 전체 데이터셋 목록 새로고침 (현재 타입과 관계없이)
            await fetchDatasetsList();
        } catch (err) {
            setError(err.message);
            setUploadProgress(null);
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
        console.log('Opening edit modal with dataset:', dataset);
        console.log('Dataset fields:', Object.keys(dataset));
        console.log('Dataset _id:', dataset._id);
        console.log('Dataset id:', dataset.id);
        console.log('Dataset did:', dataset.did);
        console.log('Dataset uid:', dataset.uid);
        console.log('Current uid:', uid);
        
        const editDataObj = { 
            ...dataset, 
            _id: dataset._id || dataset.id || dataset.did,
            uid: dataset.uid || uid,
            datasetType: dataType 
        };
        
        console.log('Final editData object:', editDataObj);
        
        setEditData(editDataObj);
        setIsEditModalOpen(true);
    };
    
    const closeEditModal = () => {
        setIsEditModalOpen(false);
        setEditData(null);
    };
    
    const openUploadModal = (dataset) => {
        console.log('Opening upload modal with dataset:', dataset);
        console.log('Dataset fields:', Object.keys(dataset));
        console.log('Dataset _id:', dataset._id);
        console.log('Dataset id:', dataset.id);
        console.log('Dataset did:', dataset.did);
        
        const uploadTargetObj = { 
            ...dataset, 
            _id: dataset._id || dataset.id || dataset.did
        };
        
        console.log('Final uploadTarget object:', uploadTargetObj);
        
        setUploadTarget(uploadTargetObj);
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
        isDeleteConfirmOpen,
        editData,
        uploadTarget,
        dataPanelTarget,
        deleteTarget,
        downloadingId,
        deletingId,
        uploadProgress,
        
        // 핸들러
        handleDownload,
        handleEdit,
        openDeleteConfirm,
        confirmDelete,
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
        setIsDeleteConfirmOpen,
        
        // 유틸리티
        fetchDatasetsList,
        refreshCurrentDatasets,
        getCurrentDatasets,
        getLimitedDatasets,
        handleShowMore,
        shouldShowMoreButton
    };
}; 