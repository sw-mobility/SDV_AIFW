import { useMemo, useCallback, useState } from 'react';
import { getParameterGroupsByAlgorithm } from '../../domain/training/parameterGroups.js';
import { validateParameter } from '../../domain/training/trainingValidation.js';
import { useTrainingCore } from './useTrainingCore.js';
import { useTrainingDatasets } from './useTrainingDatasets.js';
import { useTrainingSnapshots } from './useTrainingSnapshots.js';
import { useTrainingExecution } from './useTrainingExecution.js';
import { useTrainingUI } from './useTrainingUI.js';

export const useTrainingState = (projectId = 'P0001') => {
  const core = useTrainingCore();
  const datasets = useTrainingDatasets();
  const snapshots = useTrainingSnapshots();
  const ui = useTrainingUI();

  // Model type state 추가
  const [modelType, setModelType] = useState('pretrained');
  const [customModel, setCustomModel] = useState('');

  const trainingConfig = useMemo(() => ({
    trainingType: core.trainingType,
    selectedDataset: datasets.selectedDataset,
    selectedSnapshot: snapshots.selectedSnapshot,
    algorithm: core.algorithm,
    algoParams: core.algoParams,
    modelType: modelType,
    customModel: customModel,
    projectId: projectId // projectId 추가
  }), [core.trainingType, datasets.selectedDataset, snapshots.selectedSnapshot, core.algorithm, core.algoParams, modelType, customModel, projectId]);

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