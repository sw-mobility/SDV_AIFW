import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import AlgorithmSelector from '../../components/features/training/AlgorithmSelector.jsx';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import TrainingExecution from '../../components/features/training/TrainingExecution.jsx';
import ContinualLearningInfo from '../../components/features/training/ContinualLearningInfo.jsx';
import TrainingTypeSelector from '../../components/features/training/TrainingTypeSelector.jsx';
import ModelSelector from '../../components/features/training/ModelSelector.jsx';
import ParameterSection from '../../components/features/training/ParameterSection.jsx';
import TrainingResultList from '../../components/features/training/TrainingResultList.jsx';
import { useTrainingState } from '../../hooks';
import { TRAINING_TYPES } from '../../domain/training/trainingTypes.js';
import { uid } from '../../api/uid.js';
import styles from './TrainingPage.module.css';

const TrainingPage = () => {
  const { projectName } = useParams();
  const [actualProjectId, setActualProjectId] = useState('P0001');
  const [projectLoading, setProjectLoading] = useState(true);
  
  const {
    // Core state
    trainingType,
    setTrainingType,
    algorithm,
    algoParams,
    paramErrors,

    // Model type state
    modelType,
    setModelType,
    customModel,
    setCustomModel,

    // Dataset state
    datasets,
    selectedDataset,
    setSelectedDataset,
    datasetLoading,
    datasetError,

    // Snapshot state
    snapshots,
    selectedSnapshot,
    setSelectedSnapshot,
    editorFileStructure,
    editorFiles,

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
    isTraining,
    progress,
    status,
    logs,
    trainingResponse,

    // UI state
    openParamGroup,
    setOpenParamGroup,
    showCodeEditor,
    setShowCodeEditor,
    selectedParamKeys,

    // Computed values
    paramGroups,

    // Event handlers
    handleAlgorithmChange,
    handleAlgoParamChange,
    handleToggleParamKey,
    handleRemoveParamKey,
    handleReset,
    handleRunTraining,
  } = useTrainingState(actualProjectId);

  // Project 정보를 가져와서 실제 pid를 찾기
  useEffect(() => {
    const fetchProjectData = async () => {
      if (!projectName) {
        setActualProjectId('P0001');
        setProjectLoading(false);
        return;
      }
      
      try {
        setProjectLoading(true);
        const response = await fetch(`http://localhost:5002/projects/projects/`, {
          headers: {
            'uid': uid
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Projects API response:', data);
          
          // projectName으로 프로젝트 찾기 (예: "mynew" -> name이 "mynew"인 프로젝트)
          const project = data.find(p => p.name === projectName);
          if (project) {
            // Projects API 응답에서 pid 필드를 사용 (Training API의 pid와 매칭)
            const projectId = project.pid || 'P0001';
            setActualProjectId(projectId);
            console.log('Project found:', project.name, 'ID:', projectId);
            console.log('Full project data:', project);
          } else {
            console.warn('Project not found, using default P0001');
            setActualProjectId('P0001');
          }
        } else {
          console.warn('Failed to fetch projects, using default P0001');
          setActualProjectId('P0001');
        }
      } catch (error) {
        console.error('Error fetching project data:', error);
        setActualProjectId('P0001');
      } finally {
        setProjectLoading(false);
      }
    };
    
    fetchProjectData();
  }, [projectName]);
  
  // 디버깅을 위한 로그
  console.log('TrainingPage - projectName:', projectName);
  console.log('TrainingPage - actualProjectId:', actualProjectId);
  console.log('TrainingPage - projectLoading:', projectLoading);

  const renderDatasetSection = () => (
      <div className={styles.selectorGroup}>
        <DatasetSelector
            datasets={datasets}
            selectedDataset={selectedDataset}
            onDatasetChange={setSelectedDataset}
            datasetLoading={datasetLoading}
            datasetError={datasetError}
        />
      </div>
  );

  const renderModelSection = () => (
    <div className={styles.selectorGroup}>
      {/* 통합된 Model Selector */}
      <ModelSelector
        modelType={modelType}
        onModelTypeChange={setModelType}
        algorithm={algorithm}
        onAlgorithmChange={handleAlgorithmChange}
        customModel={customModel}
        onCustomModelChange={setCustomModel}
        projectId={actualProjectId}
        disabled={isTraining}
        projectLoading={projectLoading}
      />
    </div>
  );

  const handleRefreshTrainingResults = useCallback(() => {
    console.log('Refreshing training results...');
    // TrainingResultList 컴포넌트에서 직접 API 호출하므로
    // 여기서는 단순히 콜백만 제공
    // 실제 refresh는 TrainingResultList 컴포넌트 내부에서 처리됨
  }, []);

  const renderParameterSection = () => (
      <ParameterSection
          showCodeEditor={showCodeEditor}
          setShowCodeEditor={setShowCodeEditor}
          paramGroups={paramGroups}
          selectedParamKeys={selectedParamKeys}
          openParamGroup={openParamGroup}
          onToggleParamKey={handleToggleParamKey}
          onRemoveParamKey={handleRemoveParamKey}
          onToggleGroup={setOpenParamGroup}
          onReset={handleReset}
          snapshots={snapshots}
          selectedSnapshot={selectedSnapshot}
          setSelectedSnapshot={setSelectedSnapshot}
          editorFileStructure={editorFileStructure}
          editorFiles={editorFiles}
          codebases={codebases}
          selectedCodebase={selectedCodebase}
          setSelectedCodebase={setSelectedCodebase}
          codebaseLoading={codebaseLoading}
          codebaseError={codebaseError}
          codebaseFileStructure={codebaseFileStructure}
          codebaseFiles={codebaseFiles}
          codebaseFilesLoading={codebaseFilesLoading}
          algoParams={algoParams}
          onParamChange={handleAlgoParamChange}
          paramErrors={paramErrors}
          isTraining={isTraining}
          trainingType={trainingType}
          selectedDataset={selectedDataset}
      />
  );

  return (
      <div className={styles.pageContainer}>
        <div className={styles.container}>
          <div className={styles.pageHeader}>
            <h1 className={styles.pageTitle}>Training</h1>
            <p className={styles.pageDescription}>
              Configure your training settings and start model training with your selected dataset.
            </p>
          </div>

          <TrainingTypeSelector
              trainingType={trainingType}
              onTrainingTypeChange={setTrainingType}
          />

          {trainingType === TRAINING_TYPES.STANDARD ? (
              <>
                {renderDatasetSection()}
                {renderModelSection()}
              </>
          ) : (
              <>
                <ContinualLearningInfo />
                {renderDatasetSection()}
                {renderModelSection()}
              </>
          )}
        </div>

        {/* Dataset과 Model이 모두 선택된 후에만 표시 */}
        {selectedDataset && ((modelType === 'pretrained' && algorithm) || (modelType === 'custom' && customModel)) && (
          <>
            {/* expert mode 삼단구조 부분 */}
            <div className={showCodeEditor ? '' : styles.container}>
              {renderParameterSection()}
            </div>

            {/* parameter 필드 height 에 따라 아래로 더 내려가는 하단 컴포넌트 */}
            <div className={styles.container}>
              <TrainingExecution
                  isTraining={isTraining}
                  progress={progress}
                  onRunTraining={handleRunTraining}
                  status={status}
                  completeText="Training completed!"
                  trainingResponse={trainingResponse}
                  modelType={modelType}
                  customModel={customModel}
              />
            </div>
          </>
        )}

        {/* Training Results List - 항상 표시 */}
        <div className={styles.container}>
          <TrainingResultList onRefresh={handleRefreshTrainingResults} />
        </div>
      </div>
  );
};

export default TrainingPage;