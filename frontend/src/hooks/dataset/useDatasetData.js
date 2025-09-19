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
      
      console.log('=== FETCH DATA DEBUG ===');
      console.log('Dataset type:', dataset.datasetType || dataset.type);
      console.log('Dataset DID:', dataset.did);
      console.log('Raw API response:', res);
      console.log('Processed data:', processedData);
      console.log('Data list length:', processedData?.data_list?.length);
      if (processedData?.data_list && processedData.data_list.length > 0) {
        console.log('First data item:', processedData.data_list[0]);
        console.log('First data item keys:', Object.keys(processedData.data_list[0]));
      }
      console.log('========================');
      
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
      // selected 배열에서 실제 파일명과 경로 추출
      const target_name_list = [];
      const target_path_list = [];
      
      console.log('Selected IDs:', selected);
      console.log('Data list length:', data?.data_list?.length);
      
      selected.forEach(selectedId => {
        console.log('Processing selectedId:', selectedId);
        
        // data.data_list에서 해당 ID를 가진 항목 찾기
        let dataItem = data?.data_list?.find(item => item._id === selectedId);
        
        // 정확한 매칭이 안 되면 여러 가지 매칭 방법 시도
        if (!dataItem) {
          // 1. 파일명 부분으로 매칭 (예: 4046b4261847fad4_jpg.rf.Pu8ZmWKhSQ9UmJVGH0o4.jpg)
          const fileNamePart = selectedId.split('-')[0];
          dataItem = data?.data_list?.find(item => 
            item._id && item._id.includes(fileNamePart)
          );
          
          // 2. 여전히 못 찾으면 다른 필드들로 검색
          if (!dataItem) {
            dataItem = data?.data_list?.find(item => 
              item.file_name && item.file_name.includes(fileNamePart) ||
              item.name && item.name.includes(fileNamePart) ||
              item.path && item.path.includes(fileNamePart)
            );
          }
        }
        
        if (dataItem) {
          console.log('Found data item:', dataItem);
          
          // 파일명 추가 (가장 적절한 필드 선택)
          const fileName = dataItem.file_name || dataItem.name || dataItem._id;
          target_name_list.push(fileName);
          
          // 파일 경로 추가 (있는 경우)
          if (dataItem.file_path || dataItem.path) {
            target_path_list.push(dataItem.file_path || dataItem.path);
          }
        } else {
          console.warn('Could not find data item for selectedId:', selectedId);
          // 매칭되는 항목을 찾을 수 없으면 selectedId를 파일명으로 사용
          target_name_list.push(selectedId);
        }
      });

      console.log('=== DELETE DATA DEBUG INFO ===');
      console.log('Selected IDs:', selected);
      console.log('Target name list:', target_name_list);
      console.log('Target path list:', target_path_list);
      console.log('Dataset data:', data);
      console.log('Dataset data.data_list length:', data?.data_list?.length);
      
      // data_list의 첫 번째 항목 구조 확인
      if (data?.data_list && data.data_list.length > 0) {
        console.log('First data item structure:', data.data_list[0]);
        console.log('First data item keys:', Object.keys(data.data_list[0]));
      }
      console.log('================================');

      const result = await deleteData({ 
        uid: dataset.uid || '', 
        target_did: dataset.did,
        target_name_list: target_name_list,
        target_path_list: target_path_list
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
  }, [selected, dataset, data]);

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
      // Labeled dataset의 경우 download-data 엔드포인트 사용
      const isLabeled = dataset.datasetType === 'labeled' || dataset.type === 'labeled';
      
      console.log('=== DOWNLOAD DATASET DEBUG ===');
      console.log('Dataset type:', dataset.datasetType || dataset.type);
      console.log('Is labeled:', isLabeled);
      console.log('Dataset DID:', datasetId);
      console.log('Dataset UID:', dataset.uid);
      console.log('Dataset name:', dataset.name);
      
      if (isLabeled) {
        // Labeled dataset: 모든 파일의 경로를 가져와서 download-data 사용
        if (!data?.data_list || !Array.isArray(data.data_list)) {
          throw new Error('No data available for download');
        }
        
        console.log('Data list length:', data.data_list.length);
        console.log('First data item structure:', data.data_list[0]);
        console.log('First data item keys:', data.data_list[0] ? Object.keys(data.data_list[0]) : 'No data');
        
        const allPaths = data.data_list
          .filter(d => d.path || d.file_path || d.image_path || d.data_path || d.filePath)
          .map(d => d.path || d.file_path || d.image_path || d.data_path || d.filePath);
        
        console.log('All paths found:', allPaths);
        console.log('Paths count:', allPaths.length);
        
        if (allPaths.length === 0) {
          throw new Error('No valid data paths found');
        }
        
        await downloadDataByPaths({ 
          uid: dataset.uid, 
          target_path_list: allPaths,
          dataset_name: dataset.name
        });
      } else {
        // Raw dataset: 기존 방식 사용
        console.log('Using downloadDatasetById for raw dataset');
        await downloadDatasetById({ 
          uid: dataset.uid, 
          target_did: datasetId,
          dataset_name: dataset.name
        });
      }
      console.log('===============================');
    } catch (err) {
      console.error('Download dataset error:', err);
      setError('Download failed: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }, [dataset, data]);

  const handleDownloadSelected = useCallback(async () => {
    if (!selected.length || !data?.data_list || !Array.isArray(data.data_list) || !dataset) {
      return;
    }
    
    setDownloading(true);
    try {
      console.log('=== DOWNLOAD SELECTED DEBUG ===');
      console.log('Selected IDs:', selected);
      console.log('Dataset type:', dataset.datasetType || dataset.type);
      console.log('Dataset DID:', dataset.did);
      console.log('Dataset UID:', dataset.uid);
      console.log('Data list length:', data.data_list.length);
      console.log('First data item structure:', data.data_list[0]);
      console.log('First data item keys:', data.data_list[0] ? Object.keys(data.data_list[0]) : 'No data');
      
      // selected 배열에는 uniqueRowId가 들어있으므로, 이를 기반으로 path를 찾아야 함
      const selectedPaths = data.data_list
        .filter(d => {
          // uniqueRowId는 fileName-index 형태이므로, fileName으로 매칭
          const fileName = d.fileName || d.name || d.file_name;
          const isSelected = selected.some(selectedId => {
            // selectedId가 fileName으로 시작하는지 확인
            return selectedId.startsWith(fileName);
          });
          const hasPath = d.path || d.file_path || d.image_path || d.data_path || d.filePath;
          const actualPath = d.path || d.file_path || d.image_path || d.data_path || d.filePath;
          console.log(`File: ${fileName}, Selected: ${isSelected}, HasPath: ${hasPath}, Path: ${actualPath}`);
          return isSelected && hasPath;
        })
        .map(d => d.path || d.file_path || d.image_path || d.data_path || d.filePath);
      
      console.log('Selected paths:', selectedPaths);
      console.log('===============================');
      
      if (selectedPaths.length === 0) {
        throw new Error('No valid data paths');
      }
      
      await downloadDataByPaths({ 
        uid: dataset.uid, 
        target_path_list: selectedPaths,
        dataset_name: dataset.name
      });
    } catch (err) {
      console.error('Download selected error:', err);
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