import { uid } from './uid.js';

/**
 * IDE 코드베이스 API
 */

const API_BASE_URL = 'http://localhost:5002';

/**
 * 코드베이스 목록 조회 API
 * @returns {Promise<Array>} 코드베이스 목록
 */
export const fetchCodebases = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/IDE/codebases`, {
      headers: {
        'uid': uid
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch codebases: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching codebases:', error);
    throw error;
  }
};

/**
 * 코드베이스 조회 API
 * @param {string} cid - 코드베이스 ID
 * @returns {Promise<{tree: Array, files: Object}>}
 */
export const fetchCodebase = async (cid) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/IDE/codebase?cid=${encodeURIComponent(cid)}`,
      {
        headers: {
          'uid': uid
        }
      }
    );
    
    if (!response.ok) {
      throw new Error(`Failed to fetch codebase: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching codebase:', error);
    throw error;
  }
};

/**
 * 코드베이스 생성 API
 * @param {Object} request - create request
 * @param {Object} data - codebase data
 * @returns {Promise<Object>} 생성 결과
 */
export const createCodebase = async (request, data = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/IDE/codebase/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'uid': uid
      },
      body: JSON.stringify({
        request,
        data
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create codebase: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error creating codebase:', error);
    throw error;
  }
};

/**
 * 코드베이스 수정 API
 * @param {Object} request - 수정 요청 데이터
 * @param {Object} data - 코드베이스 데이터
 * @returns {Promise<Object>} 수정 결과
 */
export const updateCodebase = async (request, data = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/IDE/codebase/update`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'uid': uid
      },
      body: JSON.stringify({
        request,
        data
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update codebase: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error updating codebase:', error);
    throw error;
  }
};

/**
 * 코드베이스 삭제 API
 * @param {string} cid - 코드베이스 ID
 * @returns {Promise<Object>} 삭제 결과
 */
export const deleteCodebase = async (cid) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/IDE/codebase?cid=${encodeURIComponent(cid)}`,
      {
        method: 'DELETE',
        headers: {
          'uid': uid
        }
      }
    );
    
    if (!response.ok) {
      throw new Error(`Failed to delete codebase: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error deleting codebase:', error);
    throw error;
  }
};

/**
 * 코드베이스 조회 API (기존 호환성 유지)
 * @param {string} algorithm - 선택된 알고리즘 (yolo_v5, yolo_v8, yolo_v11)
 * @returns {Promise<{tree: Array, files: Object}>}
 */
export const fetchCodeTemplate = async (algorithm) => {
  try {
    // algorithm을 cid로 매핑 (yolo_v5 -> yolo, yolo_v8 -> yolo, yolo_v11 -> yolo)
    const cid = mapAlgorithmToCid(algorithm);
    
    const data = await fetchCodebase(cid);
    
    // 백엔드 응답을 프론트엔드 형식으로 변환
    return transformCodebaseResponse(data, algorithm);
  } catch (error) {
    console.error('Error fetching code template:', error);
    throw error;
  }
};

/**
 * 코드베이스 저장 API (추후 구현 예정)
 * @param {string} algorithm - 알고리즘
 * @param {string} projectId - 프로젝트 ID
 * @param {Object} files - 업데이트할 파일들
 * @returns {Promise<Object>}
 */
export const saveCodeTemplate = async (algorithm, projectId, files) => {
  try {
    // 현재는 저장 API가 없으므로 성공 응답 반환
    console.log('Save functionality not implemented yet', { algorithm, projectId, files });
    
    return {
      success: true,
      snapshotId: `snapshot_${Date.now()}`,
      algorithm,
      projectId,
      savedAt: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error saving code template:', error);
    throw error;
  }
};

/**
 * 스냅샷 저장 API
 * @param {string} uid - 사용자 ID
 * @param {Object} snapshotData - 스냅샷 데이터
 * @returns {Promise<Object>}
 */
export const saveSnapshot = async (uid, snapshotData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/IDE/snapshot`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'uid': uid
      },
      body: JSON.stringify(snapshotData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return {
      success: true,
      data: result,
      message: 'Snapshot saved successfully'
    };
  } catch (error) {
    console.error('Error saving snapshot:', error);
    return {
      success: false,
      message: error.message || 'Failed to save snapshot'
    };
  }
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


