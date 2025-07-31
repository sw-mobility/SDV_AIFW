import React from 'react';
import styles from './ValidationPage.module.css';
import ModelSelector from '../../components/features/optimization/ModelSelector.jsx';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import MetricSelector from '../../components/features/validation/MetricSelector.jsx';
import ValidationWorkspace from '../../components/features/validation/ValidationWorkspace.jsx';
import { useValidation } from '../../hooks';

const ValidationPage = () => {
  const {
    selectedModel,
    setSelectedModel,
    selectedMetric,
    setSelectedMetric,
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
    metricOptions
  } = useValidation();

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Validation</h1>
        <p className={styles.pageDescription}>
          Select a model, metric, and dataset to run validation. Results will be saved and displayed below.
        </p>
      </div>
      {error && (
        <div className={styles.errorMessage}>
          <span>Error: {error}</span>
        </div>
      )}
      <div className={styles.sectionWrap}>
        {/* 좌측: model dataset metric 선택 필드 */}
        <div className={styles.leftPanel}>
          <div className={`${styles.selectorBlock} ${styles.noBorder}`}>
            <ModelSelector 
              value={selectedModel} 
              onChange={setSelectedModel} 
              disabled={status === 'running'} 
              showInfo={false} 
            />
          </div>
          <div className={`${styles.selectorBlock} ${styles.noBorder}`}>
            <MetricSelector 
              value={selectedMetric}
              onChange={setSelectedMetric}
              disabled={status === 'running'}
              options={metricOptions}
            />
          </div>
          <div className={`${styles.selectorBlock} ${styles.noBorder}`}>
            <DatasetSelector
              datasets={datasets}
              selectedDataset={selectedDataset}
              onDatasetChange={setSelectedDataset}
              datasetLoading={datasetLoading}
              datasetError={datasetError}
            />
          </div>
        </div>
        
        {/* 우측: 실행, 결과 컴포넌트 */}
        <div className={styles.rightPanel}>
          <ValidationWorkspace
            status={status}
            progress={progress}
            onRunValidation={handleRunValidation}
            isDisabled={!selectedModel || !selectedMetric || !selectedDataset || status === 'running'}
            isRunning={status === 'running'}
            results={results}
          />
        </div>
      </div>
    </div>
  );
};

export default ValidationPage;



