import { useMemo, useCallback, useState, useEffect } from 'react';
import { getParameterGroupsByAlgorithm } from '../../domain/training/parameterGroups.js';
import { validateParameter } from '../../domain/training/trainingValidation.js';
import { useTrainingCore } from './useTrainingCore.js';
import { useTrainingDatasets } from './useTrainingDatasets.js';
import { useTrainingSnapshots } from './useTrainingSnapshots.js';
import { useTrainingExecution } from './useTrainingExecution.js';
import { useTrainingUI } from './useTrainingUI.js';
import { fetchCodebases, fetchCodebase } from '../../api/codeTemplates.js';

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

export const useTrainingState = (projectId = 'P0001') => {
  const core = useTrainingCore();
  const datasets = useTrainingDatasets();
  const snapshots = useTrainingSnapshots();
  const ui = useTrainingUI();

  // Model type state 추가
  const [modelType, setModelType] = useState('pretrained');
  const [customModel, setCustomModel] = useState('');

  // Codebase state 추가
  const [codebases, setCodebases] = useState([]);
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [codebaseLoading, setCodebaseLoading] = useState(false);
  const [codebaseError, setCodebaseError] = useState(null);
  const [codebaseFileStructure, setCodebaseFileStructure] = useState([]);
  const [codebaseFiles, setCodebaseFiles] = useState({});
  const [codebaseFilesLoading, setCodebaseFilesLoading] = useState(false);

  const trainingConfig = useMemo(() => ({
    trainingType: core.trainingType,
    selectedDataset: datasets.selectedDataset,
    selectedSnapshot: snapshots.selectedSnapshot,
    selectedCodebase: selectedCodebase,
    algorithm: core.algorithm,
    algoParams: core.algoParams,
    modelType: modelType,
    customModel: customModel,
    projectId: projectId // projectId 추가
  }), [core.trainingType, datasets.selectedDataset, snapshots.selectedSnapshot, selectedCodebase, core.algorithm, core.algoParams, modelType, customModel, projectId]);

  const execution = useTrainingExecution(trainingConfig);

  const paramGroups = useMemo(() => 
    getParameterGroupsByAlgorithm(core.algorithm), 
    [core.algorithm]
  );

  // Codebase 목록 불러오기
  useEffect(() => {
    const loadCodebases = async () => {
      try {
        setCodebaseLoading(true);
        setCodebaseError(null);
        const codebaseList = await fetchCodebases();
        setCodebases(codebaseList);
      } catch (error) {
        console.error('Failed to load codebases:', error);
        setCodebaseError('Failed to load codebases');
      } finally {
        setCodebaseLoading(false);
      }
    };

    loadCodebases();
  }, []);

  // 선택된 codebase의 파일 구조 불러오기
  useEffect(() => {
    if (!selectedCodebase) {
      // codebase가 선택되지 않았을 때는 파일 구조와 파일 내용 초기화
      setCodebaseFileStructure([]);
      setCodebaseFiles({});
      setCodebaseFilesLoading(false);
      return;
    }

    const loadCodebaseFiles = async () => {
      try {
        setCodebaseFilesLoading(true);
        const codebaseData = await fetchCodebase(selectedCodebase.cid);
        
        // codebase 데이터를 editor 형식으로 변환 (useCodeEditor의 transformCodebaseResponse 로직 사용)
        const transformedData = transformCodebaseResponse(codebaseData, selectedCodebase.algorithm || 'yolo');
        
        // codebase 파일 구조와 파일 내용 설정
        setCodebaseFileStructure(transformedData.fileStructure || []);
        setCodebaseFiles(transformedData.files || {});
      } catch (error) {
        console.error('Failed to load codebase files:', error);
      } finally {
        setCodebaseFilesLoading(false);
      }
    };

    loadCodebaseFiles();
  }, [selectedCodebase]);

  // Event handlers that combine multiple hooks
  const handleAlgorithmChange = useCallback((newAlgorithm) => {
    core.setAlgorithm(newAlgorithm);
    ui.resetUI();
  }, [core, ui]);

  const handleAlgoParamChange = useCallback((key, value, param) => {
    core.updateParam(key, value);
    const { error } = validateParameter(param, value);
    core.updateParamError(key, error);
  }, [core]);

  const handleToggleParamKey = useCallback((key) => {
    ui.toggleParamKey(key);
  }, [ui]);

  const handleRemoveParamKey = useCallback((key) => {
    ui.removeParamKey(key);
  }, [ui]);

  const handleReset = useCallback(() => {
    ui.resetUI();
    core.resetParams();
  }, [ui, core]);

  const handleRunTraining = useCallback(() => {
    execution.runTraining();
  }, [execution]);

  // Model type 변경 핸들러
  const handleModelTypeChange = useCallback((newModelType) => {
    setModelType(newModelType);
    // Model type이 변경되면 custom model 초기화
    if (newModelType === 'pretrained') {
      setCustomModel('');
    }
  }, []);

  // Custom model 변경 핸들러
  const handleCustomModelChange = useCallback((newCustomModel) => {
    setCustomModel(newCustomModel);
  }, []);

  return {
    // Core state
    trainingType: core.trainingType,
    setTrainingType: core.setTrainingType,
    algorithm: core.algorithm,
    setAlgorithm: handleAlgorithmChange,
    algoParams: core.algoParams,
    setAlgoParams: core.setAlgoParams,
    paramErrors: core.paramErrors,
    setParamErrors: core.setParamErrors,

    // Model type state
    modelType,
    setModelType: handleModelTypeChange,
    customModel,
    setCustomModel: handleCustomModelChange,

    // Dataset state
    datasets: datasets.datasets,
    selectedDataset: datasets.selectedDataset,
    setSelectedDataset: datasets.setSelectedDataset,
    datasetLoading: datasets.datasetLoading,
    datasetError: datasets.datasetError,

    // Snapshot state
    snapshots: snapshots.snapshots,
    selectedSnapshot: snapshots.selectedSnapshot,
    setSelectedSnapshot: snapshots.setSelectedSnapshot,
    editorFileStructure: snapshots.editorFileStructure,
    editorFiles: snapshots.editorFiles,

    // Codebase state
    codebases,
    selectedCodebase,
    setSelectedCodebase,
    codebaseLoading,
    codebaseError,
    codebaseFileStructure,
    codebaseFiles,
    codebaseFilesLoading,

    // Training execution state
    isTraining: execution.isRunning,
    progress: execution.progress,
    status: execution.status,
    logs: execution.logs,
    trainingResponse: execution.trainingResponse,

    // UI state
    openParamGroup: ui.openParamGroup,
    setOpenParamGroup: ui.setOpenParamGroup,
    showCodeEditor: ui.showCodeEditor,
    setShowCodeEditor: ui.setShowCodeEditor,
    selectedParamKeys: ui.selectedParamKeys,

    // Computed values
    paramGroups,

    // Event handlers
    handleAlgorithmChange,
    handleAlgoParamChange,
    handleToggleParamKey,
    handleRemoveParamKey,
    handleReset,
    handleRunTraining,
  };
}; 