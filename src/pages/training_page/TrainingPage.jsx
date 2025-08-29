import React from 'react';
import AlgorithmSelector from '../../components/features/training/AlgorithmSelector.jsx';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import TrainingExecution from '../../components/features/training/TrainingExecution.jsx';
import ContinualLearningInfo from '../../components/features/training/ContinualLearningInfo.jsx';
import TrainingTypeSelector from '../../components/features/training/TrainingTypeSelector.jsx';
import ParameterSection from '../../components/features/training/ParameterSection.jsx';
import { useTrainingState } from '../../hooks';
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
          onReset={handleReset}
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

        {/* expert mode 삼단구조 부분 */}
        <div className={styles.parameterSectionWrapper}>
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
          />
        </div>
      </div>
  );
};

export default TrainingPage;