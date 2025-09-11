import React, { useCallback } from 'react';
import styles from './LabelingPage.module.css';
import DatasetSelector from '../../components/features/training/DatasetSelector';
import LabelingFormatSelector from '../../components/features/labeling/LabelingFormatSelector';
import LabelingModelTypeSelector from '../../components/features/labeling/LabelingModelTypeSelector';
import LabelingParameterSection from '../../components/features/labeling/LabelingParameterSection';
import Button from '../../components/ui/atoms/Button';
import ProgressBar from '../../components/ui/atoms/ProgressBar';
import StatusBadge from '../../components/features/labeling/StatusBadge';
import LabeledDatasetsList from '../../components/features/labeling/LabeledDatasetsList';
import { useLabeling, useLabelingWorkspace } from '../../hooks';

const LabelingPage = () => {
  const {
    datasets,
    loading,
    error,
    selectedDataset,
    setSelectedDataset,
    datasetLoading,
    datasetError,
    labelingFormat,
    setLabelingFormat,
    modelType,
    setModelType
  } = useLabeling();

  const {
    status,
    progress,
    labelingParams,
    selectedParamKeys,
    result,
    error: labelingError,
    labeledDatasets,
    isPolling,
    handleRunLabeling,
    handleParamChange,
    resetParams,
    setSelectedParamKeys,
    fetchLabeledDatasetsList
  } = useLabelingWorkspace(selectedDataset);

  // Refresh 핸들러
  const handleRefreshLabeledDatasets = useCallback(() => {
    console.log('Refreshing labeled datasets...');
    fetchLabeledDatasetsList();
  }, [fetchLabeledDatasetsList]);

  const renderFormatAndModelTypeSection = () => (
    <div className={styles.selectorGroup} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
      <LabelingFormatSelector
        labelingFormat={labelingFormat}
        onLabelingFormatChange={setLabelingFormat}
        disabled={status === 'running'}
      />
      <LabelingModelTypeSelector
        modelType={modelType}
        onModelTypeChange={setModelType}
        disabled={status === 'running'}
      />
    </div>
  );

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
      selectedParamKeys={selectedParamKeys}
      setSelectedParamKeys={setSelectedParamKeys}
      disabled={status === 'running'}
    />
  );

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Labeling</h1>
          <p className={styles.pageDescription}>
            Select format, model type, dataset and configure parameters to start automatic labeling.
          </p>
        </div>
        
        {error && (
          <div className={styles.errorMessage}>
            <span>Error loading datasets: {error}</span>
          </div>
        )}
        
        {/* Format & Model Type 선택기 */}
        {renderFormatAndModelTypeSection()}
        
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
              disabled={!selectedDataset || status === 'running' || isPolling}
              className={styles.runButton}
            >
              {status === 'running' || isPolling ? 'Processing...' : `Run ${labelingFormat.toUpperCase()} ${modelType.toUpperCase()} Labeling`}
            </Button>
          </div>
          
          {status !== 'idle' && (
            <div className={styles.progressSection}>
              <ProgressBar
                percentage={progress}
                status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'running'}
                runningText={isPolling ? "Waiting for labeling completion..." : `${labelingFormat.toUpperCase()} ${modelType.toUpperCase()} labeling in progress...`}
                completeText="Labeling completed! New labeled dataset created."
                errorText="Labeling failed."
              />
            </div>
          )}
          
          {/* Labeled Datasets 목록 */}
          <LabeledDatasetsList 
            labeledDatasets={labeledDatasets} 
            isPolling={isPolling}
            onRefresh={handleRefreshLabeledDatasets}
            loading={false}
          />
        </div>
      </div>
    </div>
  );
};

export default LabelingPage;