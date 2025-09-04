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
    } catch (err) {
      setError(err.message);
      console.error('Failed to load codebases:', err);
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
      const result = await createCodebase(requestData);
      
      // 목록 새로고침
      await loadCodebases();
      
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
      const result = await updateCodebase({ cid, ...requestData });
      
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
