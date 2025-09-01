import { useState, useEffect, useCallback } from 'react';
import { fetchCodeTemplate, saveCodeTemplate } from '../../api/codeTemplates.js';
import { useParams } from 'react-router-dom';

/**
 * 코드 에디터를 위한 커스텀 훅
 * 알고리즘별 파일 구조와 코드를 동적으로 로드하고 관리
 */
export const useCodeEditor = (selectedAlgorithm) => {
  const { projectName } = useParams();
  
  // State
  const [fileStructure, setFileStructure] = useState([]);
  const [files, setFiles] = useState({});
  const [activeFile, setActiveFile] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState(null);

  /**
   * 선택된 알고리즘에 따라 코드 템플릿을 로드
   */
  const loadCodeTemplate = useCallback(async (algorithm) => {
    if (!algorithm) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // 실제 API 호출 (mock API 제거)
      const data = await fetchCodeTemplate(algorithm);
      
      setFileStructure(data.fileStructure || []);
      setFiles(data.files || {});
      
      // 첫 번째 파일을 활성 파일로 설정
      const firstFile = findFirstFile(data.fileStructure);
      if (firstFile) {
        setActiveFile(firstFile);
      }
      
      setHasUnsavedChanges(false);
      setLastSavedAt(data.lastModified ? new Date(data.lastModified) : null);
      
    } catch (err) {
      setError(err.message);
      console.error('Failed to load code template:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 파일 구조에서 첫 번째 파일 찾기
   */
  const findFirstFile = (structure, parentPath = '') => {
    for (const item of structure) {
      const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
      
      if (item.type === 'file') {
        return currentPath;
      } else if ((item.type === 'folder' || item.type === 'directory') && item.children) {
        const firstFile = findFirstFile(item.children, currentPath);
        if (firstFile) return firstFile;
      }
    }
    return '';
  };

  /**
   * 파일 내용 업데이트
   */
  const updateFileContent = useCallback((filename, content) => {
    setFiles(prev => ({
      ...prev,
      [filename]: {
        ...prev[filename],
        code: content
      }
    }));
    setHasUnsavedChanges(true);
  }, []);

  /**
   * 파일 언어 업데이트
   */
  const updateFileLanguage = useCallback((filename, language) => {
    setFiles(prev => ({
      ...prev,
      [filename]: {
        ...prev[filename],
        language: language
      }
    }));
    setHasUnsavedChanges(true);
  }, []);

  /**
   * 활성 파일 변경
   */
  const changeActiveFile = useCallback((filename) => {
    setActiveFile(filename);
  }, []);

  /**
   * 변경사항 저장
   */
  const saveChanges = useCallback(async () => {
    if (!selectedAlgorithm || !hasUnsavedChanges) {
      return { success: true, message: 'No changes to save' };
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // 실제 API 호출 (mock API 제거)
      const result = await saveCodeTemplate(selectedAlgorithm, projectName, files);
      
      setHasUnsavedChanges(false);
      setLastSavedAt(new Date());
      
      return {
        success: true,
        message: 'Changes saved successfully',
        snapshotId: result.snapshotId
      };
      
    } catch (err) {
      setError(err.message);
      return {
        success: false,
        message: err.message
      };
    } finally {
      setLoading(false);
    }
  }, [selectedAlgorithm, projectName, files, hasUnsavedChanges]);

  /**
   * 변경사항 폐기
   */
  const discardChanges = useCallback(() => {
    loadCodeTemplate(selectedAlgorithm);
  }, [loadCodeTemplate, selectedAlgorithm]);

  /**
   * 새 파일 생성
   */
  const createNewFile = useCallback((fileName, fileType = 'python') => {
    const language = getLanguageFromExtension(fileName);
    const defaultContent = getDefaultFileContent(fileName, language);
    
    setFiles(prev => ({
      ...prev,
      [fileName]: {
        code: defaultContent,
        language: language
      }
    }));
    
    setActiveFile(fileName);
    setHasUnsavedChanges(true);
  }, []);

  /**
   * 파일 확장자에서 언어 감지
   */
  const getLanguageFromExtension = (fileName) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    const extensionMap = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'json': 'json',
      'yaml': 'yaml',
      'yml': 'yaml',
      'txt': 'plaintext',
      'md': 'markdown',
      'sh': 'shell',
      'cfg': 'ini',
      'conf': 'ini'
    };
    return extensionMap[ext] || 'python';
  };

  /**
   * 기본 파일 내용 생성
   */
  const getDefaultFileContent = (fileName, language) => {
    if (language === 'python') {
      return `"""
${fileName}
${new Date().toISOString().split('T')[0]} - Created
"""

# TODO: Implement functionality
print("Hello from ${fileName}")
`;
    } else if (language === 'yaml') {
      return `# ${fileName}
# Configuration file

# TODO: Add configuration
`;
    } else if (language === 'json') {
      return `{
  "name": "${fileName.replace('.json', '')}",
  "version": "1.0.0",
  "description": "TODO: Add description"
}`;
    }
    return `# ${fileName}\n# TODO: Add content\n`;
  };

  // 알고리즘이 변경될 때마다 코드 템플릿 로드
  useEffect(() => {
    if (selectedAlgorithm) {
      loadCodeTemplate(selectedAlgorithm);
    }
  }, [selectedAlgorithm, loadCodeTemplate]);

  return {
    // State
    fileStructure,
    files,
    activeFile,
    loading,
    error,
    hasUnsavedChanges,
    lastSavedAt,
    
    // Actions
    updateFileContent,
    updateFileLanguage,
    changeActiveFile,
    saveChanges,
    discardChanges,
    createNewFile,
    loadCodeTemplate,
    
    // Computed
    currentFile: files[activeFile] || { code: '', language: 'python' },
    isEmpty: Object.keys(files).length === 0
  };
};
