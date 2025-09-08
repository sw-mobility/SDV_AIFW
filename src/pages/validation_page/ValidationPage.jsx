import React, { useRef, useCallback, useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import styles from './ValidationPage.module.css';
import DatasetSelector from '../../components/features/training/DatasetSelector.jsx';
import ValidationParameterSection from '../../components/features/validation/ValidationParameterSection.jsx';
import Button from '../../components/ui/atoms/Button.jsx';
import ProgressBar from '../../components/ui/atoms/ProgressBar.jsx';
import StatusBadge from '../../components/features/validation/StatusBadge.jsx';

import ValidationHistoryList from '../../components/features/validation/ValidationHistoryList.jsx';
import { useValidation } from '../../hooks';
import { uid } from '../../api/uid.js';

const ValidationPage = () => {
  const { projectName } = useParams();
  const [actualProjectId, setActualProjectId] = useState('P0001');
  const [projectLoading, setProjectLoading] = useState(true);

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
    resetValidationParams,
    refreshValidationHistory,
    // Codebase 관련 상태 추가
    codebases,
    selectedCodebase,
    setSelectedCodebase,
    codebaseLoading,
    codebaseError,
    codebaseFileStructure,
    codebaseFiles,
    codebaseFilesLoading,
    showCodeEditor,
    setShowCodeEditor
  } = useValidation(actualProjectId);

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
          
          // projectName으로 프로젝트 찾기
          const project = data.find(p => p.name === projectName);
          if (project) {
            const projectId = project.pid || 'P0001';
            setActualProjectId(projectId);
            console.log('Project found:', project.name, 'ID:', projectId);
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
      projectId={actualProjectId}
      // Codebase 관련 props 추가 (training과 동일)
      codebases={codebases}
      selectedCodebase={selectedCodebase}
      setSelectedCodebase={setSelectedCodebase}
      codebaseLoading={codebaseLoading}
      codebaseError={codebaseError}
      codebaseFileStructure={codebaseFileStructure}
      codebaseFiles={codebaseFiles}
      codebaseFilesLoading={codebaseFilesLoading}
      showCodeEditor={showCodeEditor}
      setShowCodeEditor={setShowCodeEditor}
      isValidating={status === 'running'}
    />
  );

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Validation</h1>
          <p className={styles.pageDescription}>
            Validate your trained models with selected datasets and analyze performance metrics.
          </p>
        </div>

        {renderDatasetSection()}
      </div>

      {/* Parameter Section - 삼단구조일 때는 container 없이, 아닐 때는 container 적용 */}
      <div className={showCodeEditor ? '' : styles.container}>
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
          

        </div>
      </div>

      {/* Validation History List - 항상 표시 */}
      <div className={`${styles.container} ${styles.historyContainer}`}>
        <ValidationHistoryList onRefresh={refreshValidationHistory} projectId={actualProjectId} />
      </div>
    </div>
  );
};

export default ValidationPage;



