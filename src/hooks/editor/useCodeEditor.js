import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchCodebase, updateCodebase } from '../../api/codeTemplates.js';

/**
 * 코드 에디터를 위한 커스텀 훅
 * 코드베이스 기반으로 파일 구조와 코드를 동적으로 로드하고 관리
 */
export const useCodeEditor = (selectedCodebase) => {
  // State
  const [fileStructure, setFileStructure] = useState([]);
  const [files, setFiles] = useState({});
  const [activeFile, setActiveFile] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [lastSavedAt, setLastSavedAt] = useState(null);

  // 최신 files 상태를 참조하기 위한 ref
  const filesRef = useRef(files);
  
  // files 상태가 변경될 때마다 ref 업데이트
  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  /**
   * 선택된 코드베이스에 따라 코드 템플릿을 로드
   */
  const loadCodeTemplate = useCallback(async (codebase) => {
    if (!codebase || !codebase.cid) return;

    setLoading(true);
    setError(null);

    try {
      // 코드베이스 조회
      const data = await fetchCodebase(codebase.cid);

      // 백엔드 응답을 프론트엔드 형식으로 변환
      const transformedData = transformCodebaseResponse(data, codebase.algorithm || 'yolo');

      setFileStructure(transformedData.fileStructure || []);
      const newFiles = transformedData.files || {};
      setFiles(newFiles);

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
   * 파일 내용 업데이트 (수정된 버전)
   */
  const updateFileContent = useCallback((filename, content) => {
    setFiles(prevFiles => {
      const newFiles = {
        ...prevFiles,
        [filename]: {
          ...prevFiles[filename],
          code: content
        }
      };
      return newFiles;
    });

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
  const changeActiveFile = useCallback((filename, onFileChangeStart, onFileChangeEnd) => {
    // 파일 변경 시작 신호
    if (onFileChangeStart) onFileChangeStart();
    
    setActiveFile(filename);
    
    // 파일 변경 완료 신호 (약간의 지연 후)
    setTimeout(() => {
      if (onFileChangeEnd) onFileChangeEnd();
    }, 50);
  }, [files]);

  /**
   * 변경사항 저장 (수정된 버전)
   */
  const saveChanges = useCallback(async (codebaseCid = null) => {
    if (!selectedCodebase || !hasUnsavedChanges) {
      return { success: true, message: 'No changes to save' };
    }

    setLoading(true);
    setError(null);

    try {
      // 코드베이스 업데이트 API 호출 - API 명세에 맞게 cid 포함
      const cid = codebaseCid || selectedCodebase.cid;
      const requestData = {
        cid: cid,
        name: selectedCodebase.name || `${selectedCodebase.algorithm || 'yolo'} Template`,
        algorithm: selectedCodebase.algorithm || 'yolo',
        stage: selectedCodebase.stage || 'training',
        task_type: selectedCodebase.task_type || 'detection',
        description: selectedCodebase.description || `Updated ${selectedCodebase.algorithm || 'yolo'} codebase`
      };

      // 현재 수정된 파일 내용을 백엔드 형식으로 변환
      // filesRef를 사용하여 최신 상태 보장
      const backendFiles = {};
      const currentFiles = filesRef.current; // ref를 통해 최신 상태 참조

      Object.entries(currentFiles).forEach(([treeFilePath, fileData]) => {
        // 백엔드에서 기대하는 원본 경로 사용 (fileData.path가 있으면 사용, 없으면 treeFilePath 사용)
        const backendPath = fileData.path || treeFilePath;
        backendFiles[backendPath] = fileData.code || '';
      });

      // 백엔드가 기대하는 구조: { tree: [...], files: {...} }
      const backendData = {
        tree: fileStructure, // 파일 구조 정보 포함
        files: backendFiles
      };

      const result = await updateCodebase(requestData, backendData);


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
  }, [selectedCodebase, hasUnsavedChanges, fileStructure]); // fileStructure dependency 추가

  /**
   * 변경사항 폐기
   */
  const discardChanges = useCallback(() => {
    loadCodeTemplate(selectedCodebase);
  }, [loadCodeTemplate, selectedCodebase]);

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

  // 코드베이스가 변경될 때마다 코드 템플릿 로드
  useEffect(() => {
    if (selectedCodebase) {
      loadCodeTemplate(selectedCodebase);
    }
  }, [selectedCodebase, loadCodeTemplate]);

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
    currentFile: (() => {
      let result = files[activeFile];
      
      // 정확한 키로 파일을 찾지 못한 경우 대체 키 검색
      if (!result && activeFile) {
        // 파일명 기반 검색
        const fileName = activeFile.split('/').pop();
        const alternativeKeys = Object.keys(files).filter(key => {
          const keyFileName = key.split('/').pop();
          return keyFileName === fileName;
        });
        
        if (alternativeKeys.length === 1) {
          result = files[alternativeKeys[0]];
        } else if (alternativeKeys.length > 1) {
          // 여러 후보가 있으면 가장 유사한 것 선택
          let bestKey = alternativeKeys[0];
          let bestScore = -1;
          
          for (const key of alternativeKeys) {
            const score = calculatePathSimilarity(activeFile, key);
            if (score > bestScore) {
              bestScore = score;
              bestKey = key;
            }
          }
          
          result = files[bestKey];
        }
      }
      
      // 최종 fallback
      if (!result) {
        result = { code: '', language: 'python' };
      }
      
      return result;
    })(),
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

  // 파일 트리에서 모든 파일 경로를 수집
  const allFilePaths = new Set();
  const collectFilePaths = (items, parentPath = '') => {
    items.forEach(item => {
      const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
      
      if (item.type === 'file') {
        allFilePaths.add(currentPath);
      } else if ((item.type === 'folder' || item.type === 'directory') && item.children) {
        collectFilePaths(item.children, currentPath);
      }
    });
  };
  collectFilePaths(fileStructure);

  // files 객체를 프론트엔드 형식으로 변환
  if (backendData.files) {
    Object.entries(backendData.files).forEach(([filePath, content]) => {
      const fileName = getFileNameFromPath(filePath);
      const language = getLanguageFromFileName(fileName);

      // 파일 트리의 경로와 매칭되는 키를 찾기
      let matchingKey = filePath;
      
      // 1. 백엔드 파일 경로가 파일 트리 경로와 정확히 일치하는지 확인
      if (allFilePaths.has(filePath)) {
        matchingKey = filePath;
        // console.log(`✅ Exact match: "${filePath}"`);
      } else {
        // console.log(`❌ No exact match for: "${filePath}", searching alternatives...`);
        
        // 2. 파일명 기반 매칭 (같은 파일명을 가진 모든 경로 찾기)
        const possibleMatches = Array.from(allFilePaths).filter(treePath => {
          const treeFileName = treePath.split('/').pop();
          return treeFileName === fileName;
        });
        
        // console.log(`🔍 Possible matches for "${fileName}":`, possibleMatches);
        
        if (possibleMatches.length === 1) {
          // 유일한 매칭이 있으면 사용
          matchingKey = possibleMatches[0];
          // console.log(`✅ Unique match found: "${matchingKey}"`);
        } else if (possibleMatches.length > 1) {
          // 여러 매칭이 있으면 백엔드 경로와 가장 유사한 것 선택
          let bestMatch = possibleMatches[0];
          let bestScore = -1;
          
          for (const treePath of possibleMatches) {
            // 경로가 정확히 일치하거나 백엔드 경로가 트리 경로의 일부인 경우 우선순위
            if (treePath === filePath) {
              bestMatch = treePath;
              bestScore = 1000; // 최고 점수
              break;
            } else if (filePath.endsWith(treePath) || treePath.endsWith(filePath)) {
              const score = Math.max(filePath.length, treePath.length);
              if (score > bestScore) {
                bestScore = score;
                bestMatch = treePath;
              }
            } else {
              const score = calculatePathSimilarity(filePath, treePath);
              if (score > bestScore) {
                bestScore = score;
                bestMatch = treePath;
              }
            }
          }
          matchingKey = bestMatch;
          // console.log(`🎯 Best match selected: "${matchingKey}" (score: ${bestScore})`);
        } else {
          // 매칭이 없으면 경고하고 원본 사용
          // console.warn(`⚠️ No matches found for "${fileName}", using original path: "${filePath}"`);
          matchingKey = filePath;
        }
      }

      transformedFiles[matchingKey] = {
        code: content,
        language: language,
        path: filePath, // 원본 백엔드 경로 정보 보존
        name: fileName  // 파일명 정보 보존
      };
    });
  }

  // 간단한 로깅 (필요시 주석 해제)
  // console.log('🔍 File mapping completed:', Object.keys(transformedFiles).length, 'files');

  return {
    algorithm,
    fileStructure,
    files: transformedFiles,
    lastModified: new Date().toISOString(),
    version: '1.0.0'
  };
};

/**
 * 두 경로의 유사도를 계산
 * @param {string} path1 - 첫 번째 경로
 * @param {string} path2 - 두 번째 경로
 * @returns {number} 유사도 점수
 */
const calculatePathSimilarity = (path1, path2) => {
  const parts1 = path1.split('/');
  const parts2 = path2.split('/');
  let commonParts = 0;
  
  const minLength = Math.min(parts1.length, parts2.length);
  for (let i = 0; i < minLength; i++) {
    if (parts1[i] === parts2[i]) {
      commonParts++;
    } else {
      break;
    }
  }
  
  return commonParts;
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