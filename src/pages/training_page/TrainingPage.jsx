import React from 'react';
import AlgorithmSelector from '../../components/training/AlgorithmSelector.jsx';
import DatasetSelector from '../../components/training/DatasetSelector.jsx';
import TrainingExecution from '../../components/training/TrainingExecution.jsx';
import ContinualLearningInfo from '../../components/training/ContinualLearningInfo.jsx';
import TrainingTypeSelector from '../../components/training/TrainingTypeSelector.jsx';
import ParameterSection from '../../components/training/ParameterSection.jsx';
import { useTrainingState } from '../../hooks/useTrainingState.js';
import { TRAINING_TYPES } from '../../domain/training/trainingTypes.js';
import styles from './TrainingPage.module.css';

const TrainingPage = () => {
  const {
    // Core state
    trainingType,
    setTrainingType,
    algorithm,
    algoParams,
    paramErrors,

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

    // Training execution state
    isTraining,
    progress,
    status,
    logs,

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
    handleRunTraining,
  } = useTrainingState();

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
          snapshots={snapshots}
          selectedSnapshot={selectedSnapshot}
          setSelectedSnapshot={setSelectedSnapshot}
          editorFileStructure={editorFileStructure}
          editorFiles={editorFiles}
          algoParams={algoParams}
          onParamChange={handleAlgoParamChange}
          paramErrors={paramErrors}
          isTraining={isTraining}
          trainingType={trainingType}
      />
  );

  return (
      <div className={styles.pageContainer}>
        {/* 상단 부분 - container 제한 */}
        <div className={styles.container}>
          {/* Page Header */}
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

          <AlgorithmSelector
              algorithm={algorithm}
              onAlgorithmChange={handleAlgorithmChange}
          />

          {trainingType === TRAINING_TYPES.STANDARD ? (
              <>
                {renderDatasetSection()}
              </>
          ) : (
              <>
                <ContinualLearningInfo />
                {renderDatasetSection()}
              </>
          )}
        </div>

        {/* 삼단구조 부분 - 동적 높이 조절 가능 */}
        <div className={styles.parameterSectionWrapper}>
          {renderParameterSection()}
        </div>

        {/* 하단 부분 - container 제한 */}
        <div className={styles.container}>
          <TrainingExecution
              isTraining={isTraining}
              progress={progress}
              logs={logs}
              onRunTraining={handleRunTraining}
              status={status}
              completeText="Training completed!"
          />
        </div>
      </div>
  );
};

export default TrainingPage;