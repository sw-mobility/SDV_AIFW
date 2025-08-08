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
        did: dataset.did ,
        id: dataset._id ,
        uid: dataset.uid || '' 
      });
      
      setData(res);
    } catch (err) {
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
    setSelected(prev => prev.includes(row._id) ? prev.filter(x => x !== row._id) : [...prev, row._id]);
  }, []);

  const handleSelectAll = useCallback(() => {
    if (!data?.data_list || !Array.isArray(data.data_list)) return;
    if (selected.length === data.data_list.length) {
      setSelected([]);
    } else {
      setSelected(data.data_list.map(d => d._id).filter(Boolean));
    }
  }, [data?.data_list, selected.length]);

  // 삭제 처리
  const handleDelete = useCallback(async () => {
    if (!selected.length || !dataset) return;
    
    setShowDeleteConfirm(false);
    try {
      await deleteData({ 
        uid: dataset.uid || '', 
        id: dataset._id, 
        target_id_list: selected 
      });
      setSelected([]);
      setRefreshKey(k => k + 1);
    } catch (err) {
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
            id: dataset._id,
            onProgress: (progress) => {
              // 진행률 업데이트
              setUploadProgress(progress);
            }
          });
        } else {
          await uploadLabeledFiles({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            id: dataset._id
          });
        }
      } else {
        if (useBatchUpload) {
          await uploadRawFilesInBatches({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            id: dataset._id,
            onProgress: (progress) => {
              // 진행률 업데이트
              setUploadProgress(progress);
            }
          });
        } else {
          await uploadRawFiles({ 
            files: uploadFiles, 
            uid: dataset.uid || '', 
            id: dataset._id
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
    if (!dataset?._id || !dataset?.uid) return;
    
    setDownloading(true);
    try {
      await downloadDatasetById({ 
        uid: dataset.uid, 
        target_id: dataset._id,
        dataset_name: dataset.name || data?.name
      });
    } catch (err) {
      setError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }, [dataset, data?.name]);

  const handleDownloadSelected = useCallback(async () => {
    if (!selected.length || !data?.data_list || !Array.isArray(data.data_list) || !dataset) return;
    
    setDownloading(true);
    try {
      const selectedPaths = data.data_list
        .filter(d => d && selected.includes(d._id) && d.path)
        .map(d => d.path);
      
      if (selectedPaths.length === 0) {
        throw new Error('No valid data paths');
      }
      
      await downloadDataByPaths({ 
        uid: dataset.uid, 
        target_path_list: selectedPaths,
        dataset_name: dataset.name || data?.name
      });
    } catch (err) {
      setError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }, [selected, data?.data_list, dataset, data?.name]);
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