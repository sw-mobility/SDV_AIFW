import React from 'react';
import styles from './ValidationPage.module.css';
import ModelSelector from '../../components/features/optimization/ModelSelector.jsx';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import ValidationWorkspace from '../../components/features/validation/ValidationWorkspace.jsx';
import { useValidation } from '../../hooks';

const ValidationPage = () => {
  const {
    selectedModel,
    setSelectedModel,
    selectedDataset,
    setSelectedDataset,
    status,
    progress,
    results,
    loading,
    error,
    datasets,
    datasetLoading,
    datasetError,
    handleRunValidation,
    validationParams,
    updateValidationParams,
    mockModels
  } = useValidation();

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Validation</h1>
        <p className={styles.pageDescription}>
          Select a model and dataset to run validation. Results will be saved and displayed below.
        </p>
      </div>
      {error && (
        <div className={styles.errorMessage}>
          <span>Error: {error}</span>
        </div>
      )}
      
      {/* 상단: Model과 Dataset 선택기 */}
      <div className={styles.selectorSection}>
        <div className={styles.selectorRow}>
          <div className={styles.selectorItem}>
            <ModelSelector 
              value={selectedModel} 
              onChange={setSelectedModel} 
              disabled={status === 'running'} 
              showInfo={false}
              models={mockModels}
            />
          </div>
          <div className={styles.selectorItem}>
            <DatasetSelector
              datasets={datasets}
              selectedDataset={selectedDataset}
              onDatasetChange={setSelectedDataset}
              datasetLoading={datasetLoading}
              datasetError={datasetError}
            />
          </div>
        </div>
      </div>
      
      {/* 하단: Validation Workspace */}
      <div className={styles.workspaceSection}>
        <ValidationWorkspace
          status={status}
          progress={progress}
          onRunValidation={handleRunValidation}
          isDisabled={!selectedModel || !selectedDataset || status === 'running'}
          isRunning={status === 'running'}
          results={results}
          validationParams={validationParams}
          onParametersChange={updateValidationParams}
        />
      </div>
    </div>
  );
};

export default ValidationPage;



