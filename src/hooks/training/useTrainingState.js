import { useMemo, useCallback } from 'react';
import { getParameterGroupsByAlgorithm } from '../../domain/training/parameterGroups.js';
import { validateParameter } from '../../domain/training/trainingValidation.js';
import { useTrainingCore } from './useTrainingCore.js';
import { useTrainingDatasets } from './useTrainingDatasets.js';
import { useTrainingSnapshots } from './useTrainingSnapshots.js';
import { useTrainingExecution } from './useTrainingExecution.js';
import { useTrainingUI } from './useTrainingUI.js';

export const useTrainingState = () => {
  const core = useTrainingCore();
  const datasets = useTrainingDatasets();
  const snapshots = useTrainingSnapshots();
  const ui = useTrainingUI();

  const trainingConfig = useMemo(() => ({
    trainingType: core.trainingType,
    selectedDataset: datasets.selectedDataset,
    selectedSnapshot: snapshots.selectedSnapshot,
    algorithm: core.algorithm,
    algoParams: core.algoParams
  }), [core.trainingType, datasets.selectedDataset, snapshots.selectedSnapshot, core.algorithm, core.algoParams]);

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
    setIsTraining: execution.start,
    progress: execution.progress,
    setProgress: execution.updateProgress,
    status: execution.status,
    setStatus: execution.start,
    logs: execution.logs,
    setLogs: execution.addLog,

    // UI state
    openParamGroup: ui.openParamGroup,
    setOpenParamGroup: ui.setOpenParamGroup,
    showCodeEditor: ui.showCodeEditor,
    setShowCodeEditor: ui.setShowCodeEditor,
    selectedParamKeys: ui.selectedParamKeys,
    setSelectedParamKeys: ui.setSelectedParamKeys,

    // Computed values
    paramGroups,
    trainingConfig,

    // Event handlers
    handleAlgorithmChange,
    handleAlgoParamChange,
    handleToggleParamKey,
    handleRemoveParamKey,
    handleReset,
    handleRunTraining,
  };
}; 