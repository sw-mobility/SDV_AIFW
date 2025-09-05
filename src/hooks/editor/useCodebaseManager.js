import { useState, useEffect, useCallback } from 'react';
import { 
  fetchCodebases, 
  fetchCodebase, 
  createCodebase, 
  updateCodebase, 
  deleteCodebase 
} from '../../api/codeTemplates.js';

/**
 * 코드베이스 관리를 위한 커스텀 훅
 * 코드베이스 CRUD 작업과 상태 관리
 */
export const useCodebaseManager = () => {
  const [codebases, setCodebases] = useState([]);
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * 코드베이스 목록 조회
   */
  const loadCodebases = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchCodebases();
      
      // updated_at 기준으로 내림차순 정렬 (최근 수정된 것이 위에)
      const sortedData = data.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0);
        const dateB = new Date(b.updated_at || b.created_at || 0);
        return dateB - dateA; // 내림차순
      });
      
      setCodebases(sortedData);
      return sortedData; // 데이터 반환
    } catch (err) {
      setError(err.message);
      console.error('Failed to load codebases:', err);
      return []; // 에러 시 빈 배열 반환
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 특정 코드베이스 조회
   */
  const loadCodebase = useCallback(async (cid) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchCodebase(cid);
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Failed to load codebase:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 새 코드베이스 생성
   */
  const handleCreateCodebase = useCallback(async (requestData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🚀 Starting codebase creation with data:', requestData);
      
      // 코드베이스 생성 (cid는 백엔드에서 자동 생성)
      // createCodebase는 (request, data) 두 개의 매개변수를 받음
      const result = await createCodebase(requestData, {});
      
      console.log('✅ Codebase creation result:', result);
      
      // 목록 새로고침하여 생성된 코드베이스의 실제 cid 확인
      const updatedCodebases = await loadCodebases();
      
      // 생성된 코드베이스 찾기 (가장 최근에 생성된 것)
      const sortedCodebases = updatedCodebases.sort((a, b) => {
        const dateA = new Date(a.created_at || 0);
        const dateB = new Date(b.created_at || 0);
        return dateB - dateA;
      });
      
      const createdCodebase = sortedCodebases[0]; // 가장 최근 생성된 코드베이스
      
      if (createdCodebase) {
        // 생성된 코드베이스에 새로 생성됨 플래그 추가
        const codebaseWithFlag = {
          ...createdCodebase,
          _isNewlyCreated: true
        };
        
        // 생성된 코드베이스 자동 선택
        setSelectedCodebase(codebaseWithFlag);
        
        // 자동으로 yolo 템플릿을 저장하여 완전한 사이클 완성
        setTimeout(async () => {
          try {
            // yolo 템플릿 데이터를 가져와서 자동 저장
            const yoloTemplateData = await fetchCodebase('yolo');
            if (yoloTemplateData && yoloTemplateData.files) {
              const updateRequest = {
                cid: createdCodebase.cid, // 실제 생성된 cid 사용
                name: requestData.name,
                algorithm: requestData.algorithm,
                stage: requestData.stage,
                task_type: requestData.task_type,
                description: requestData.description
              };
              await updateCodebase(updateRequest, yoloTemplateData);
              console.log('Auto-saved yolo template to new codebase with cid:', createdCodebase.cid);
              
              // 목록 새로고침하여 저장된 상태 반영
              await loadCodebases();
              
              // 자동 저장 완료 후 플래그 제거
              setSelectedCodebase(prev => ({
                ...prev,
                _isNewlyCreated: false
              }));
            }
          } catch (error) {
            console.error('Failed to auto-save yolo template:', error);
          }
        }, 1000); // 1초 후 자동 저장
      }
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Failed to create codebase:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadCodebases]);

  /**
   * 코드베이스 수정
   */
  const handleUpdateCodebase = useCallback(async (cid, requestData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🚀 Starting codebase update with cid:', cid, 'data:', requestData);
      
      // updateCodebase는 (request, data) 두 개의 매개변수를 받음
      const result = await updateCodebase({ cid, ...requestData }, {});
      
      console.log('✅ Codebase update result:', result);
      
      // 목록 새로고침
      await loadCodebases();
      
      // 현재 선택된 코드베이스가 수정된 것이라면 업데이트
      if (selectedCodebase?.cid === cid) {
        setSelectedCodebase(prev => ({
          ...prev,
          ...requestData
        }));
      }
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Failed to update codebase:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadCodebases, selectedCodebase]);

  /**
   * 코드베이스 삭제
   */
  const handleDeleteCodebase = useCallback(async (cid) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await deleteCodebase(cid);
      
      // 목록 새로고침
      await loadCodebases();
      
      // 삭제된 코드베이스가 현재 선택된 것이라면 선택 해제
      if (selectedCodebase?.cid === cid) {
        setSelectedCodebase(null);
      }
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Failed to delete codebase:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadCodebases, selectedCodebase]);

  /**
   * 코드베이스 선택
   */
  const handleSelectCodebase = useCallback((codebase) => {
    setSelectedCodebase(codebase);
  }, []);

  /**
   * 코드베이스 선택 해제
   */
  const handleDeselectCodebase = useCallback(() => {
    setSelectedCodebase(null);
  }, []);

  // 컴포넌트 마운트 시 코드베이스 목록 로드
  useEffect(() => {
    loadCodebases();
  }, [loadCodebases]);

  return {
    // State
    codebases,
    selectedCodebase,
    loading,
    error,
    
    // Actions
    loadCodebases,
    loadCodebase,
    handleCreateCodebase,
    handleUpdateCodebase,
    handleDeleteCodebase,
    handleSelectCodebase,
    handleDeselectCodebase,
    
    // Computed
    hasCodebases: codebases.length > 0,
    isCodebaseSelected: selectedCodebase !== null
  };
};
