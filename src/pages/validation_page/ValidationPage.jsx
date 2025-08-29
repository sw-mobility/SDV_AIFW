import React from 'react';
import styles from './ValidationPage.module.css';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import ValidationParameterSection from '../../components/features/validation/ValidationParameterSection.jsx';
import Button from '../../components/ui/atoms/Button.jsx';
import ProgressBar from '../../components/ui/atoms/ProgressBar.jsx';
import StatusBadge from '../../components/features/validation/StatusBadge.jsx';
import ResultsTable from '../../components/features/validation/ResultsTable.jsx';
import { useValidation } from '../../hooks';

const ValidationPage = () => {
  const {
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
    resetValidationParams
  } = useValidation();

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
    <ValidationParameterSection
      validationParams={validationParams}
      onParamChange={updateValidationParams}
      onReset={resetValidationParams}
      disabled={status === 'running'}
    />
  );

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Validation</h1>
          <p className={styles.pageDescription}>
            Select a dataset to run validation. The model will be automatically determined from the dataset.
          </p>
        </div>
        
        {error && (
          <div className={styles.errorMessage}>
            <span>Error: {error}</span>
          </div>
        )}
        
        {/* Dataset 선택기 */}
        {renderDatasetSection()}
      </div>

      {/* 파라미터 섹션 - training 페이지와 동일한 구조 */}
      <div className={styles.parameterSectionWrapper}>
        {renderParameterSection()}
      </div>

      {/* 하단: 실행 및 결과 섹션 */}
      <div className={styles.container}>
        <div className={styles.executionSection}>
          <div className={styles.executionHeader}>
            <div className={styles.executionTitle}>
              <h2 style={{ fontSize: 22, marginBottom: 0 }}>Validation Execution</h2>
            </div>
            <StatusBadge status={status} />
          </div>
          
          <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="primary"
              size="medium"
              onClick={handleRunValidation}
              disabled={!selectedDataset || status === 'running'}
              className={styles.runButton}
            >
              {status === 'running' ? 'Running...' : 'Run Validation'}
            </Button>
          </div>
          
          {status !== 'idle' && (
            <div className={styles.progressSection}>
              <ProgressBar
                percentage={progress}
                status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
                runningText="Validation in progress..."
                completeText="Validation completed!"
                errorText="Validation failed."
              />
            </div>
          )}
          
          {/* 결과 섹션 */}
          {results.length > 0 && (
            <div className={styles.resultsSection}>
              <h3 style={{ marginBottom: 16 }}>Results</h3>
              <ResultsTable results={results} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ValidationPage;



