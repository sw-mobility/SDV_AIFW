import React, { useEffect } from 'react';
import CodeEditor from '../../components/common/CodeEditor.jsx';
import TabNavigation from '../../components/common/TabNavigation.jsx';
import AlgorithmSelector from '../../components/training/AlgorithmSelector.jsx';
import DatasetSelector from '../../components/training/DatasetSelector.jsx';
import SnapshotSelector from '../../components/training/SnapshotSelector.jsx';
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
import { ChevronDown, ChevronUp, Info, X } from 'lucide-react';

const TrainingPage = () => {
  const {
    // Training Type & Mode
    trainingType, setTrainingType,
    mode,

  // Dataset
    datasets,
    selectedDataset, setSelectedDataset,
    datasetLoading,
    datasetError,

  // Snapshot
    snapshots,
    selectedSnapshot, setSelectedSnapshot,
    
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
    selectedParamKeys, setSelectedParamKeys,
    editorFileStructure,
    editorFiles
  } = useTrainingState();

  // Training 타입 탭 설정
  const trainingTabs = [
    { value: 'standard', label: 'Standard Training' },
    { value: 'continual', label: 'Continual Training' }
  ];

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
    setSelectedParamKeys([]);
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

  // 파라미터 선택/해제 토글
  const handleToggleParamKey = (key) => {
    setSelectedParamKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  // 칩에서 X 클릭 시 해제
  const handleRemoveParamKey = (key) => {
    setSelectedParamKeys((prev) => prev.filter((k) => k !== key));
  };

  // 현재 파라미터 그룹 가져오기
  const paramGroups = getParameterGroups(algorithm);

  // progress 100%시 status를 success로 자동 변경
  useEffect(() => {
    if (progress === 100 && status !== 'success') {
      setStatus('success');
    }
  }, [progress, status, setStatus]);

  return (
    <div className={styles.container}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#222", marginBottom: "10px"}}>Training</div>
      <div className={styles.pageDescription}>
        Configure your training settings and start model training with your selected dataset.
      </div>
      
      {/* Training Type Tabs */}
      <TabNavigation
        tabs={trainingTabs}
        activeTab={trainingType}
        onTabChange={setTrainingType}
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
            {/* Left: Param Accordion List + 선택 칩 리스트 */}
            <div className={styles.paramSummaryBox + ' ' + styles.sectionCard}>
              <div className={styles.paramGroupTitle} style={{ fontSize: 17, marginBottom: 12 }}>Parameters</div>
              {/* 선택된 파라미터 칩 리스트 */}
              {selectedParamKeys.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
                  {selectedParamKeys.map((key) => {
                    let foundParam = null;
                    let isProjectInfo = false;
                    for (const group of paramGroups) {
                      for (const param of group.params) {
                        if (param.key === key) {
                          foundParam = param;
                          isProjectInfo = group.group === 'Project Information';
                          break;
                        }
                      }
                      if (foundParam) break;
                    }
                    if (!foundParam) return null;
                    return (
                      <span
                        key={key}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          background: isProjectInfo ? '#dcfce7' : '#eff6ff',
                          color: isProjectInfo ? '#166534' : '#2563eb',
                          borderRadius: 16,
                          padding: '4px 12px',
                          fontWeight: 500,
                          fontSize: 13,
                          boxShadow: isProjectInfo ? '0 1px 4px rgba(34,197,94,0.08)' : '0 1px 4px rgba(59,130,246,0.08)',
                          cursor: 'pointer',
                        }}
                      >
                        {foundParam.label}
                        <X size={16} style={{ marginLeft: 4, cursor: 'pointer' }} onClick={() => handleRemoveParamKey(key)} />
                      </span>
                    );
                  })}
            </div>
              )}
              {/* 아코디언 그룹별로 렌더링 */}
              {paramGroups.map((group, gidx) => {
                // Project Information 그룹이면 model_version, model_size, task_type만 노출
                let paramsToShow = group.params;
                if (group.group === 'Project Information') {
                  paramsToShow = group.params.filter(p => ['model_version', 'model_size', 'task_type'].includes(p.key));
                }
                if (paramsToShow.length === 0) return null;
                return (
                  <div key={group.group} className={styles.accordionCard}>
                    <div
                      className={styles.accordionHeader + ' ' + (openParamGroup === gidx ? styles.accordionOpen : '')}
                      onClick={() => setOpenParamGroup(openParamGroup === gidx ? -1 : gidx)}
                      tabIndex={0}
                      role="button"
                      aria-expanded={openParamGroup === gidx}
                    >
                      <span>{group.group}</span>
                      <span className={styles.accordionArrow}>{openParamGroup === gidx ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                    </div>
                    {openParamGroup === gidx && (
                      <div className={styles.accordionContent}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                          {paramsToShow.map((param) => {
                            const isSelected = selectedParamKeys.includes(param.key);
                            const isProjectInfo = group.group === 'Project Information';
                            return (
                              <button
                                key={param.key}
                                type="button"
                                onClick={() => handleToggleParamKey(param.key)}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  border: 'none',
                                  outline: 'none',
                                  borderRadius: 16,
                                  padding: '7px 16px',
                                  fontWeight: 500,
                                  fontSize: 14,
                                  marginBottom: 6,
                                  background: isSelected 
                                    ? (isProjectInfo ? '#22c55e' : '#2563eb') 
                                    : '#f1f5f9',
                                  color: isSelected ? '#fff' : '#222',
                                  boxShadow: isSelected 
                                    ? (isProjectInfo ? '0 2px 8px rgba(34,197,94,0.10)' : '0 2px 8px rgba(37,99,235,0.10)') 
                                    : 'none',
                                  cursor: 'pointer',
                                  transition: 'all 0.18s',
                                }}
                              >
                                {param.label}
                                {param.desc && (
                                  <Info size={15} color={isSelected ? '#fff' : '#888'} style={{ marginLeft: 6 }} title={param.desc} />
                                )}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
                </div>
            {/* Right: 선택된 파라미터만 편집 */}
            <div className={styles.paramCardWrap}>
              {selectedParamKeys.length === 0 ? (
                <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
                  <span style={{ color: '#aaa', fontSize: 15 }}>왼쪽에서 파라미터를 선택하세요.</span>
                </div>
              ) : (
                selectedParamKeys.map((key) => {
                  // 파라미터 정의 찾기
                  let foundParam = null;
                  for (const group of paramGroups) {
                    for (const param of group.params) {
                      if (param.key === key) foundParam = param;
                    }
                  }
                  if (!foundParam) return null;
                  return (
                    <ParameterEditor
                      key={key}
                      currentParam={foundParam}
                      algoParams={algoParams}
                      onParamChange={handleAlgoParamChange}
                      paramErrors={paramErrors}
                      isTraining={isTraining}
                    />
                  );
                })
              )}
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
          
          {/* Parameters - Only show 'Training Parameters' group for Continual */}
          <div className={styles.paramCardWrap}>
            {paramGroups.filter(g => g.group === 'Training Parameters').map((group, idx) => (
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
        status={status}
        completeText="Training completed!"
      />
    </div>
  );
};

export default TrainingPage;
