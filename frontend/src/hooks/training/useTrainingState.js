import { useMemo, useCallback, useState, useEffect } from 'react';
import { getParameterGroupsByAlgorithm } from '../../domain/training/parameterGroups.js';
import { validateParameter } from '../../domain/training/trainingValidation.js';
import { useTrainingCore } from './useTrainingCore.js';
import { useTrainingDatasets } from './useTrainingDatasets.js';
import { useTrainingExecution } from './useTrainingExecution.js';
import { useTrainingUI } from './useTrainingUI.js';
import { fetchCodebases, fetchCodebase } from '../../api/codeTemplates.js';

export const useTrainingState = (projectId = 'P0001') => {
  const core = useTrainingCore();
  const datasets = useTrainingDatasets();
  const ui = useTrainingUI();

  // Model type state 추가
  const [modelType, setModelType] = useState('pretrained');
  const [customModel, setCustomModel] = useState('');
  
  // Training format state 추가
  const [trainingFormat, setTrainingFormat] = useState('yolo');

  // Codebase 관련 상태 (validation과 동일하게 직접 관리)
  const [codebases, setCodebases] = useState([]);
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [codebaseLoading, setCodebaseLoading] = useState(false);
  const [codebaseError, setCodebaseError] = useState(null);
  const [codebaseFileStructure, setCodebaseFileStructure] = useState([]);
  const [codebaseFiles, setCodebaseFiles] = useState({});
  const [codebaseFilesLoading, setCodebaseFilesLoading] = useState(false);


  // Codebase 관련 함수들 (validation과 동일)
  const fetchCodebasesList = useCallback(async () => {
    setCodebaseLoading(true);
    setCodebaseError(null);
    try {
      const data = await fetchCodebases();
      const sortedData = data.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0);
        const dateB = new Date(b.updated_at || b.created_at || 0);
        return dateB - dateA;
      });
      setCodebases(sortedData);
    } catch (err) {
      setCodebaseError(err.message);
      console.error('Failed to load codebases:', err);
    } finally {
      setCodebaseLoading(false);
    }
  }, []);

  const fetchCodebaseFiles = useCallback(async (codebaseId) => {
    setCodebaseFilesLoading(true);
    try {
      const codebaseData = await fetchCodebase(codebaseId);
      
      setCodebaseFileStructure(codebaseData.tree || []);
      setCodebaseFiles(codebaseData.files || {});
    } catch (err) {
      console.error('Failed to load codebase files:', err);
      setCodebaseFileStructure([]);
      setCodebaseFiles({});
    } finally {
      setCodebaseFilesLoading(false);
    }
  }, []);

  // selectedCodebase가 변경될 때 파일 정보 로드
  useEffect(() => {
    if (selectedCodebase?.cid) {
      fetchCodebaseFiles(selectedCodebase.cid);
    } else {
      setCodebaseFileStructure([]);
      setCodebaseFiles({});
    }
  }, [selectedCodebase, fetchCodebaseFiles]);

  // 초기 codebase 목록 로드
  useEffect(() => {
    fetchCodebasesList();
  }, [fetchCodebasesList]);

  const trainingConfig = useMemo(() => ({
    trainingType: core.trainingType,
    selectedDataset: datasets.selectedDataset,
    selectedCodebase: selectedCodebase, // codebase 추가
    algorithm: core.algorithm,
    algoParams: core.algoParams,
    modelType: modelType,
    customModel: customModel,
    projectId: projectId // projectId 추가
  }), [core.trainingType, datasets.selectedDataset, selectedCodebase, core.algorithm, core.algoParams, modelType, customModel, projectId]);

  const execution = useTrainingExecution(trainingConfig);

  const paramGroups = useMemo(() => 
    getParameterGroupsByAlgorithm(core.algorithm), 
    [core.algorithm]
  );

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
    
    // Training format state
    trainingFormat,
    setTrainingFormat,

    // Dataset state
    datasets: datasets.datasets,
    selectedDataset: datasets.selectedDataset,
    setSelectedDataset: datasets.setSelectedDataset,
    datasetLoading: datasets.datasetLoading,
    datasetError: datasets.datasetError,


    // Codebase state (validation과 동일)
    codebases,
    selectedCodebase,
    setSelectedCodebase,
    codebaseLoading,
    codebaseError,
    codebaseFileStructure,
    codebaseFiles,
    codebaseFilesLoading,

    // Training execution state (validation과 동일한 구조)
    isTraining: execution.isRunning,
    progress: execution.progress,
    status: execution.status,
    logs: execution.logs,
    error: execution.error,
    trainingResponse: execution.trainingResponse,
    refreshTrainingHistory: execution.refreshTrainingHistory,

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