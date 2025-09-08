import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import styles from './OptimizationPage.module.css';
import OptimizationTypeSelector from '../../components/features/optimization/OptimizationTypeSelector.jsx';
import OptimizationModelSelector from '../../components/features/optimization/OptimizationModelSelector.jsx';
import OptimizationParameterSection from '../../components/features/optimization/OptimizationParameterSection.jsx';
import OptimizationExecution from '../../components/features/optimization/OptimizationExecution.jsx';
import OptimizationHistoryList from '../../components/features/optimization/OptimizationHistoryList.jsx';
import { useOptimizationState } from '../../hooks';
import { uid } from '../../api/uid.js';

const OptimizationPage = () => {
  const { projectName } = useParams();
  const [actualProjectId, setActualProjectId] = useState('P0001');
  const [projectLoading, setProjectLoading] = useState(true);

  const {
    // Core state
    optimizationType,
    modelType,
    modelId,
    optimizationParams,
    
    // Execution state
    isRunning,
    progress,
    status,
    error,
    
    // Event handlers
    handleRunOptimization,
    handleOptimizationTypeChange,
    handleModelTypeChange,
    handleModelIdChange,
    handleParamChange,
    resetOptimization,
    refreshOptimizationHistory,
    refreshModelList,
    setRefreshModelListCallback
  } = useOptimizationState(actualProjectId);

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

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Optimization</h1>
          <p className={styles.pageDescription}>
            Optimize your trained models through conversion, pruning, and analysis. Select a model and optimization type to get started.
          </p>
        </div>
        

        
        {/* Error Message */}
        {error && (
          <div className={styles.errorMessage}>
            <span>Error: {error}</span>
          </div>
        )}

        {/* Selectors */}
        <div className={styles.selectorGroup}>
          <OptimizationTypeSelector
            optimizationType={optimizationType}
            onOptimizationTypeChange={handleOptimizationTypeChange}
            disabled={isRunning}
          />
        </div>

        {/* Model Selection */}
        <OptimizationModelSelector
          selectedModelType={modelType}
          selectedModelId={modelId}
          onModelTypeChange={handleModelTypeChange}
          onModelIdChange={handleModelIdChange}
          optimizationType={optimizationType}
          disabled={isRunning}
          setRefreshCallback={setRefreshModelListCallback}
          projectId={actualProjectId}
        />

        {/* Parameter Configuration - Optimization Type 선택 후에만 표시 (CHECK MODEL STATS 제외) */}
        {optimizationType && optimizationType !== 'check_model_stats' && (
          <OptimizationParameterSection
            optimizationType={optimizationType}
            optimizationParams={optimizationParams}
            onParamChange={handleParamChange}
            onReset={resetOptimization}
            isRunning={isRunning}
            projectId={actualProjectId}
          />
        )}
        
        {/* Execution - Optimization Type 선택 후에만 표시 */}
        {optimizationType && (
          <OptimizationExecution
            isRunning={isRunning}
            progress={progress}
            status={status}
            onRunOptimization={handleRunOptimization}
            optimizationType={optimizationType}
            optimizationParams={optimizationParams}
          />
        )}
      </div>

      {/* Optimization History List - 항상 표시 */}
      <div className={`${styles.container} ${styles.historyContainer}`}>
        <OptimizationHistoryList onRefresh={refreshOptimizationHistory} projectId={actualProjectId} />
      </div>
    </div>
  );
};

export default OptimizationPage;
