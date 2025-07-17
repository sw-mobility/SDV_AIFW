import React from 'react';
import CodeEditor from '../../components/common/CodeEditor.jsx';
import TrainingTypeTabs from '../../components/training/TrainingTypeTabs.jsx';
import AlgorithmSelector from '../../components/training/AlgorithmSelector.jsx';
import DatasetSelector from '../../components/training/DatasetSelector.jsx';
import SnapshotSelector from '../../components/training/SnapshotSelector.jsx';
import ParameterChipList from '../../components/training/ParameterChipList.jsx';
import ParameterEditor from '../../components/training/ParameterEditor.jsx';
import TrainingExecution from '../../components/training/TrainingExecution.jsx';
import ContinualLearningInfo from '../../components/training/ContinualLearningInfo.jsx';
import { useTrainingState } from '../../hooks/useTrainingState.js';
import { 
  validateParam, 
  validateTrainingExecution, 
  executeTraining, 
  getParameterGroups 
} from '../../services/trainingService.js';
import styles from './TrainingPage.module.css';

const TrainingPageRefactored = () => {
  const {
    // Training Type & Mode
    trainingType, setTrainingType,
    mode, setMode,
    
    // Dataset
    datasets, setDatasets,
    selectedDataset, setSelectedDataset,
    datasetLoading, setDatasetLoading,
    datasetError, setDatasetError,
    
    // Snapshot
    snapshots, setSnapshots,
    selectedSnapshot, setSelectedSnapshot,
    snapshotModalOpen, setSnapshotModalOpen,
    
    // Algorithm
    algorithm, setAlgorithm,
    algoParams, setAlgoParams,
    paramErrors, setParamErrors,
    
    // Training
    isTraining, setIsTraining,
    progress, setProgress,
    status, setStatus,
    logs, setLogs,
    
    // UI State
    openParamGroup, setOpenParamGroup,
    showCodeEditor, setShowCodeEditor,
    selectedParamKey, setSelectedParamKey,
    editorFileStructure, setEditorFileStructure,
    editorFiles, setEditorFiles,
  } = useTrainingState();

  // 파라미터 변경 핸들러
  const handleAlgoParamChange = (key, value, param) => {
    setAlgoParams(p => ({ ...p, [key]: value }));
    const { error } = validateParam(param, value);
    setParamErrors(prev => ({ ...prev, [key]: error }));
  };

  // 알고리즘 변경 핸들러
  const handleAlgorithmChange = (newAlgorithm) => {
    setAlgorithm(newAlgorithm);
    setAlgoParams({});
    setOpenParamGroup(0);
    setSelectedParamKey(null);
  };

  // Training 실행 핸들러
  const handleRunTraining = async () => {
    const validation = validateTrainingExecution(trainingType, selectedDataset, selectedSnapshot, mode);
    
    if (!validation.isValid) {
      alert(validation.errors.join('\n'));
      return;
    }

    setIsTraining(true);
    setStatus('Training');
    setLogs([]);
    setProgress(0);

    try {
      const result = await executeTraining({
        trainingType,
        selectedDataset,
        selectedSnapshot,
        algorithm,
        algoParams
      });
      
      if (result.success) {
        setLogs(l => [...l, result.message]);
      }
    } catch (error) {
      setLogs(l => [...l, `Error: ${error.message}`]);
      setIsTraining(false);
    }
  };

  // Drawer close handler
  const handleCloseDrawer = () => setShowCodeEditor(false);

  // 현재 파라미터 그룹 가져오기
  const paramGroups = getParameterGroups(algorithm);

  // 현재 선택된 파라미터 가져오기
  const getCurrentParam = () => {
    for (const group of paramGroups) {
      for (const param of group.params) {
        if (param.key === selectedParamKey) return { ...param, group: group.group };
      }
    }
    return null;
  };

  const currentParam = getCurrentParam();

  return (
    <div className={styles.container}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#222", marginBottom: "10px"}}>Training</div>
      
      {/* Training Type Tabs */}
      <TrainingTypeTabs 
        trainingType={trainingType} 
        onTrainingTypeChange={setTrainingType} 
      />
      
      {/* Algorithm Selector */}
      <AlgorithmSelector 
        algorithm={algorithm}
        onAlgorithmChange={handleAlgorithmChange}
        onShowCodeEditor={() => setShowCodeEditor(true)}
      />
      
      {/* Standard/Continual UI */}
      {trainingType === 'standard' ? (
        <>
          {/* Dataset & Snapshot (Standard) */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <DatasetSelector 
                datasets={datasets}
                selectedDataset={selectedDataset}
                onDatasetChange={setSelectedDataset}
                datasetLoading={datasetLoading}
                datasetError={datasetError}
              />
              <SnapshotSelector 
                snapshots={snapshots}
                selectedSnapshot={selectedSnapshot}
                onSnapshotChange={setSelectedSnapshot}
              />
            </div>
          </div>
          
          {/* Parameters & Summary 2단 레이아웃 */}
          <div className={styles.paramSectionWrap}>
            {/* Left: ParamChipList */}
            <div className={styles.paramSummaryBox + ' ' + styles.sectionCard}>
              <div className={styles.paramGroupTitle} style={{ fontSize: 17, marginBottom: 12 }}>Parameters</div>
              <ParameterChipList
                paramGroups={paramGroups}
                algoParams={algoParams}
                selectedKey={selectedParamKey}
                onSelect={setSelectedParamKey}
              />
            </div>
            
            {/* Right: Parameter Form */}
            <div className={styles.paramCardWrap}>
              <ParameterEditor
                currentParam={currentParam}
                algoParams={algoParams}
                onParamChange={handleAlgoParamChange}
                paramErrors={paramErrors}
                isTraining={isTraining}
              />
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Continual Learning Info */}
          <ContinualLearningInfo />
          
          {/* Base Snapshot (required) and New Dataset */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <SnapshotSelector 
                snapshots={snapshots}
                selectedSnapshot={selectedSnapshot}
                onSnapshotChange={setSelectedSnapshot}
                isRequired={true}
              />
              <DatasetSelector 
                datasets={datasets}
                selectedDataset={selectedDataset}
                onDatasetChange={setSelectedDataset}
                datasetLoading={datasetLoading}
                datasetError={datasetError}
              />
            </div>
          </div>
          
          {/* Parameters - Only show 'Training' group for Continual */}
          <div className={styles.paramCardWrap}>
            {paramGroups.filter(g => g.group === 'Training').map((group, idx) => (
              <div key={group.group} className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === idx ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === idx ? -1 : idx)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === idx}
                >
                  <span>{group.group}</span>
                  <span className={styles.accordionArrow}>
                    {openParamGroup === idx ? '▼' : '▶'}
                  </span>
                </div>
                {openParamGroup === idx && (
                  <div className={styles.accordionContent}>
                    {group.params.map(param => (
                      <div className={styles.paramRow} key={param.key}>
                        <label className={styles.paramLabel}>{param.label}</label>
                        <ParameterEditor
                          currentParam={param}
                          algoParams={algoParams}
                          onParamChange={handleAlgoParamChange}
                          paramErrors={paramErrors}
                          isTraining={isTraining}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Drawer for Code Editor */}
      {showCodeEditor && (
        <>
          <div className={styles.drawerOverlay} onClick={handleCloseDrawer}></div>
          <div className={styles.codeDrawer}>
            <div className={styles.drawerEditorWrap}>
              <CodeEditor
                snapshotName={selectedSnapshot ? selectedSnapshot.name : 'Default Snapshot'}
                fileStructure={editorFileStructure}
                files={editorFiles}
                onSaveSnapshot={name => {
                  alert(`Saved as snapshot: ${name}`);
                }}
                onCloseDrawer={handleCloseDrawer}
              />
            </div>
          </div>
        </>
      )}
      
      {/* Training Execution */}
      <TrainingExecution 
        isTraining={isTraining}
        progress={progress}
        logs={logs}
        onRunTraining={handleRunTraining}
      />
    </div>
  );
};

export default TrainingPageRefactored; 