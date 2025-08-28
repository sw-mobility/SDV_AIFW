import { useState, useEffect, useCallback } from 'react';
import { 
    getRawDataset, 
    getLabeledDataset, 
    deleteData, 
    uploadRawFiles, 
    uploadLabeledFiles,
    uploadRawFilesInBatches,
    uploadLabeledFilesInBatches,
    downloadDatasetById, 
    downloadDataByPaths 
} from '../../api/datasets.js';

export const useDatasetData = (dataset, isOpen = false) => {
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
  const [uploadProgress, setUploadProgress] = useState(null);

  // 데이터 조회
  const fetchData = useCallback(async () => {
    if (!dataset) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const isLabeled = dataset.datasetType === 'labeled' || dataset.type === 'labeled';
      const fetchFunction = isLabeled ? getLabeledDataset : getRawDataset;
      
      const res = await fetchFunction({ 
        id: dataset._id || dataset.id || dataset.did,
        uid: dataset.uid || '' 
      });
      
      // API 응답 구조에 따라 데이터 변환
      let processedData;
      if (res.dataset && res.data) {
        // 응답이 {dataset: {...}, data: Array} 구조인 경우
        processedData = {
          ...res.dataset,
          data_list: res.data || []
        };
      } else if (res.data_list) {
        // 응답이 직접 {name, description, data_list, ...} 구조인 경우
        processedData = res;
      } else {
        // 기타 경우
        processedData = res;
      }
      
      setData(processedData);
    } catch (err) {
      console.error('Error fetching dataset data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset, refreshKey]);

  // 초기 로드 및 refresh
  useEffect(() => {
    if (!isOpen || !dataset) {
      // 모달이 닫히거나 dataset이 없을 때 상태 초기화
      setData(null);
      setError(null);
      setLoading(false);
      setSelected([]);
      setUploadFiles([]);
      setUploadError(null);
      setShowDeleteConfirm(false);
      setDownloading(false);
      return;
    }
    fetchData();
  }, [fetchData, isOpen, dataset]);

  // 선택 관리
  const handleSelect = useCallback((row) => {
    if (!row || !row._id) return;
    setSelected(prev => {
      const isSelected = prev.includes(row._id);
      if (isSelected) {
        return prev.filter(x => x !== row._id);
      } else {
        return [...prev, row._id];
      }
    });
  }, []);

  const handleSelectAll = useCallback(() => {
    if (!data?.data_list || !Array.isArray(data.data_list)) return;
    
    const validIds = data.data_list.map(d => d._id).filter(Boolean);
    const allSelected = validIds.every(id => selected.includes(id));
    
    if (allSelected) {
      setSelected([]);
    } else {
      setSelected(validIds);
    }
  }, [data?.data_list, selected]);

  // 삭제 처리
  const handleDelete = useCallback(async () => {
    if (!selected.length || !dataset) return;
    
    setShowDeleteConfirm(false);
    try {
      // selected 배열에서 파일명만 추출 (uniqueRowId에서 fileName-index 형태에서 fileName만 추출)
      const target_name_list = selected.map(id => {
        // uniqueRowId가 fileName-index 형태인 경우 fileName만 추출
        if (id.includes('-')) {
          const parts = id.split('-');
          const lastPart = parts[parts.length - 1];
          // 마지막 부분이 숫자인지 확인
          if (!isNaN(lastPart)) {
            return parts.slice(0, -1).join('-');
          }
        }
        return id;
      });

      const result = await deleteData({ 
        uid: dataset.uid || '', 
        target_did: dataset._id,
        target_name_list: target_name_list
      });
      
      console.log('Delete data result:', result);
      
      setSelected([]);
      setRefreshKey(k => k + 1);
    } catch (err) {
      console.error('Delete data error:', err);
      
      // "삭제된 문서가 없습니다" 메시지 처리
      if (err.message.includes('삭제된 문서가 없습니다') || err.message.includes('No documents found to delete')) {
        console.log('No documents to delete - treating as success');
        // 에러를 표시하지 않고 성공으로 처리
        setSelected([]);
        setRefreshKey(k => k + 1);
        return;
      }
      
      // 404 에러나 resources not found 메시지 처리
      if (err.message.includes('resources not found') || err.message.includes('404')) {
        console.log('Resources not found - treating as success');
        // 에러를 표시하지 않고 성공으로 처리
        setSelected([]);
        setRefreshKey(k => k + 1);
        return;
      }
      
      setError(err.message);
    }
  }, [selected, dataset]);

  // 업로드 처리
  const handleUpload = useCallback(async (e) => {
    if (e) e.preventDefault();
    if (!uploadFiles.length || !dataset) return;
    
    setUploading(true);
    setUploadError(null);
    setUploadProgress(null);
    
    try {
      const isLabeled = dataset.datasetType === 'labeled' || dataset.type === 'labeled';
      
      // 1000개 이상의 파일인 경우 배치 업로드 사용
      const useBatchUpload = uploadFiles.length > 1000;
      
      if (isLabeled) {
        if (useBatchUpload) {
          await uploadLabeledFilesInBatches({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            did: dataset._id,
            onProgress: (progress) => {
              // 진행률 업데이트
              setUploadProgress(progress);
            }
          });
        } else {
          await uploadLabeledFiles({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            did: dataset._id
          });
        }
      } else {
        if (useBatchUpload) {
          await uploadRawFilesInBatches({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            did: dataset._id,
            onProgress: (progress) => {
              // 진행률 업데이트
              setUploadProgress(progress);
            }
          });
        } else {
          await uploadRawFiles({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            did: dataset._id
          });
        }
      }
      
      setUploadFiles([]);
      setUploadProgress(null);
      
      // 업로드 완료 후 잠시 대기 후 데이터 새로고침
      setTimeout(() => {
        setRefreshKey(k => k + 1);
        console.log('Dataset data refreshed after upload');
      }, 1000); // 1초 대기
    } catch (err) {
      setUploadError(err.message);
      setUploadProgress(null);
    } finally {
      setUploading(false);
    }
  }, [uploadFiles, dataset]);

  // 다운로드 처리
  const handleDownloadDataset = useCallback(async () => {
    // dataset 객체에서 did 필드를 사용
    const datasetId = dataset?.did;
    if (!datasetId || !dataset?.uid) {
      setError('Download failed: Missing dataset ID or UID');
      return;
    }
    
    setDownloading(true);
    try {
      await downloadDatasetById({ 
        uid: dataset.uid, 
        target_did: datasetId,
        dataset_name: dataset.name
      });
    } catch (err) {
      setError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }, [dataset]);

  const handleDownloadSelected = useCallback(async () => {
    if (!selected.length || !data?.data_list || !Array.isArray(data.data_list) || !dataset) {
      return;
    }
    
    setDownloading(true);
    try {
      // selected 배열에는 uniqueRowId가 들어있으므로, 이를 기반으로 path를 찾아야 함
      const selectedPaths = data.data_list
        .filter(d => {
          // uniqueRowId는 fileName-index 형태이므로, fileName으로 매칭
          const fileName = d.fileName || d.name;
          const isSelected = selected.some(selectedId => {
            // selectedId가 fileName으로 시작하는지 확인
            return selectedId.startsWith(fileName);
          });
          const hasPath = d.path;
          return isSelected && hasPath;
        })
        .map(d => d.path);
      
      if (selectedPaths.length === 0) {
        throw new Error('No valid data paths');
      }
      
      await downloadDataByPaths({ 
        uid: dataset.uid, 
        target_path_list: selectedPaths,
        dataset_name: dataset.name
      });
    } catch (err) {
      setError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }, [selected, data?.data_list, dataset]);
  // 파일 업데이트
  const updateUploadFiles = useCallback((newFiles) => {
    setUploadFiles(newFiles);
    setUploadError(null);
  }, []);

  // 삭제 확인 토글
  const toggleDeleteConfirm = useCallback(() => {
    setShowDeleteConfirm(prev => !prev);
  }, []);

  return {
    data,
    loading,
    error,
    selected,
    setSelected,
    uploading,
    uploadError,
    uploadFiles,
    showDeleteConfirm,
    downloading,
    uploadProgress,
    refreshKey, // refreshKey 추가

    handleSelect,
    handleSelectAll,
    handleDelete,
    handleUpload,
    handleDownloadDataset,
    handleDownloadSelected,
    updateUploadFiles,
    toggleDeleteConfirm,

    isLabeled: dataset && (dataset.datasetType === 'labeled' || dataset.type === 'labeled')
  };
}; 