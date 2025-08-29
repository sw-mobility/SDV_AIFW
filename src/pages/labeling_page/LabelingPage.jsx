import React from 'react';
import styles from './LabelingPage.module.css';
import DatasetSelector from '../../components/features/training/DatasetSelector';
import LabelingParameterSection from '../../components/features/labeling/LabelingParameterSection';
import Button from '../../components/ui/atoms/Button';
import ProgressBar from '../../components/ui/atoms/ProgressBar';
import StatusBadge from '../../components/features/labeling/StatusBadge';
import ResultsTable from '../../components/features/labeling/ResultsTable';
import { useLabeling, useLabelingWorkspace } from '../../hooks';

const LabelingPage = () => {
  const {
    datasets,
    loading,
    error,
    selectedDataset,
    setSelectedDataset,
    datasetLoading,
    datasetError
  } = useLabeling();

  const {
    status,
    progress,
    labelingParams,
    result,
    error: labelingError,
    handleRunLabeling,
    handleParamChange,
    resetParams,
  } = useLabelingWorkspace(selectedDataset);

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
    <LabelingParameterSection
      labelingParams={labelingParams}
      onParamChange={handleParamChange}
      onReset={resetParams}
      disabled={status === 'running'}
    />
  );

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Labeling</h1>
          <p className={styles.pageDescription}>
            Select a dataset and configure YOLO labeling parameters to start automatic labeling.
          </p>
        </div>
        
        {error && (
          <div className={styles.errorMessage}>
            <span>Error loading datasets: {error}</span>
          </div>
        )}
        
        {/* Dataset 선택기 */}
        {renderDatasetSection()}
      </div>

      {/* 파라미터 섹션 - validation 페이지와 동일한 구조 */}
      <div className={styles.parameterSectionWrapper}>
        {renderParameterSection()}
      </div>

      {/* 하단: 실행 및 결과 섹션 */}
      <div className={styles.container}>
        <div className={styles.executionSection}>
          <div className={styles.executionHeader}>
            <div className={styles.executionTitle}>
              <h2 style={{ fontSize: 22, marginBottom: 0 }}>Labeling Execution</h2>
            </div>
            <StatusBadge status={status} />
          </div>
          
          {labelingError && (
            <div className={styles.errorMessage}>
              <span>Error: {labelingError}</span>
            </div>
          )}
          
          <div style={{ margin: '32px 0 24px 0', display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="primary"
              size="medium"
              onClick={handleRunLabeling}
              disabled={!selectedDataset || status === 'running'}
              className={styles.runButton}
            >
              {status === 'running' ? 'Running...' : 'Run YOLO Labeling'}
            </Button>
          </div>
          
          {status !== 'idle' && (
            <div className={styles.progressSection}>
              <ProgressBar
                percentage={progress}
                status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
                runningText="YOLO labeling in progress..."
                completeText="Labeling completed!"
                errorText="Labeling failed."
              />
            </div>
          )}
          
          {/* 결과 섹션 */}
          {result && status === 'success' && (
            <div className={styles.resultsSection}>
              <h3 style={{ marginBottom: 16 }}>Results</h3>
              <ResultsTable result={result} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LabelingPage;