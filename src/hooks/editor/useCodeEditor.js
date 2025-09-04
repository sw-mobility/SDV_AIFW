import { useState, useEffect, useCallback } from 'react';
import { fetchCodebase, updateCodebase } from '../../api/codeTemplates.js';
import { useParams } from 'react-router-dom';

/**
 * 코드 에디터를 위한 커스텀 훅
 * 코드베이스 기반으로 파일 구조와 코드를 동적으로 로드하고 관리
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
   * 선택된 알고리즘에 따라 코드 템플릿을 로드 (기존 호환성 유지)
   */
  const loadCodeTemplate = useCallback(async (algorithm) => {
    if (!algorithm) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // 알고리즘을 cid로 매핑하여 코드베이스 조회
      const cid = mapAlgorithmToCid(algorithm);
      const data = await fetchCodebase(cid);
      
      // 백엔드 응답을 프론트엔드 형식으로 변환
      const transformedData = transformCodebaseResponse(data, algorithm);
      
      setFileStructure(transformedData.fileStructure || []);
      setFiles(transformedData.files || {});
      
      // 첫 번째 파일을 활성 파일로 설정
      const firstFile = findFirstFile(transformedData.fileStructure);
      if (firstFile) {
        setActiveFile(firstFile);
      }
      
      setHasUnsavedChanges(false);
      setLastSavedAt(transformedData.lastModified ? new Date(transformedData.lastModified) : null);
      
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
  const saveChanges = useCallback(async (codebaseCid = null) => {
    if (!selectedAlgorithm || !hasUnsavedChanges) {
      return { success: true, message: 'No changes to save' };
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // 코드베이스 업데이트 API 호출 - API 명세에 맞게 cid 포함
      const cid = codebaseCid || mapAlgorithmToCid(selectedAlgorithm);
      const requestData = {
        cid: cid,
        name: `${selectedAlgorithm} Template`,
        algorithm: selectedAlgorithm,
        stage: 'training',
        task_type: 'object_detection',
        description: `Updated ${selectedAlgorithm} codebase`
      };
      
      const result = await updateCodebase(requestData, { files });
      
      setHasUnsavedChanges(false);
      setLastSavedAt(new Date());
      
      return {
        success: true,
        message: 'Changes saved successfully',
        data: result
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
  }, [selectedAlgorithm, files, hasUnsavedChanges]);

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

/**
 * 알고리즘 이름을 cid로 매핑
 * @param {string} algorithm - 프론트엔드 알고리즘 이름
 * @returns {string} 백엔드 cid
 */
const mapAlgorithmToCid = (algorithm) => {
  const algorithmToCidMap = {
    'yolo_v5': 'yolo',
    'yolo_v8': 'yolo', 
    'yolo_v11': 'yolo'
  };
  
  return algorithmToCidMap[algorithm] || 'yolo';
};

/**
 * 백엔드 응답을 프론트엔드 형식으로 변환
 * @param {Object} backendData - 백엔드 응답 데이터
 * @param {string} algorithm - 알고리즘 이름
 * @returns {Object} 프론트엔드 형식 데이터
 */
const transformCodebaseResponse = (backendData, algorithm) => {
  // 백엔드 응답: { tree: [...], files: { "path": "content" } }
  // 프론트엔드 기대: { fileStructure: [...], files: { "filename": { code: "...", language: "..." } } }
  
  const fileStructure = backendData.tree || [];
  const transformedFiles = {};
  
  // files 객체를 프론트엔드 형식으로 변환
  if (backendData.files) {
    Object.entries(backendData.files).forEach(([filePath, content]) => {
      // 파일 경로에서 파일명만 추출 (중복 방지를 위해 전체 경로를 키로 사용)
      const fileName = getFileNameFromPath(filePath);
      const language = getLanguageFromFileName(fileName);
      
      // 모든 파일에 대해 전체 경로를 키로 사용하여 중복 방지
      const fileKey = filePath;
      
      transformedFiles[fileKey] = {
        code: content,
        language: language,
        path: filePath, // 원본 경로 정보 보존
        name: fileName  // 파일명 정보 보존
      };
    });
  }
  
  return {
    algorithm,
    fileStructure,
    files: transformedFiles,
    lastModified: new Date().toISOString(),
    version: '1.0.0'
  };
};

/**
 * 파일 경로에서 파일명 추출
 * @param {string} filePath - 파일 경로
 * @returns {string} 파일명
 */
const getFileNameFromPath = (filePath) => {
  return filePath.split('/').pop() || filePath;
};

/**
 * 파일명에서 언어 감지
 * @param {string} fileName - 파일명
 * @returns {string} 프로그래밍 언어
 */
const getLanguageFromFileName = (fileName) => {
  const ext = fileName.split('.').pop()?.toLowerCase();
  const extensionMap = {
    'py': 'python', 'pyc': 'python', 'pyo': 'python', 'pyz': 'python', 'js': 'javascript', 'ts': 'typescript', 'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'txt': 'plaintext',
    'md': 'markdown',
    'sh': 'shell',
    'cfg': 'ini',
    'conf': 'ini',
    'html': 'html',
    'css': 'css',
    'cpp': 'cpp',
    'c': 'c',
    'java': 'java'
  };
  
  return extensionMap[ext] || 'plaintext';
};
