import React, { useState } from 'react';
import styles from './OptimizationPage.module.css';
import OptimizationTypeSelector from '../../components/features/optimization/OptimizationTypeSelector.jsx';
import OptimizationParameterSection from '../../components/features/optimization/OptimizationParameterSection.jsx';
import OptimizationExecution from '../../components/features/optimization/OptimizationExecution.jsx';
import { useOptimizationState } from '../../hooks';

const OptimizationPage = () => {
  const [projectId, setProjectId] = useState('P0001');
  
  const {
    // Core state
    optimizationType,
    optimizationParams,
    
    // Execution state
    isRunning,
    progress,
    status,
    results,
    error,
    
    // Event handlers
    handleRunOptimization,
    handleOptimizationTypeChange,
    handleParamChange,
    resetOptimization
  } = useOptimizationState();

  // Project ID 변경 시 optimizationParams 업데이트
  const handleProjectIdChange = (newProjectId) => {
    setProjectId(newProjectId);
    handleParamChange('pid', newProjectId);
  };

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Optimization</h1>
          <p className={styles.pageDescription}>
            Optimize your trained models through conversion, pruning, and analysis. Select a model and optimization type to get started.
          </p>
        </div>
        
        {/* Project ID Selector */}
        <div className={styles.selectorGroup}>
          <div className={styles.projectSelector}>
            <label htmlFor="projectId">Project ID:</label>
            <input
              id="projectId"
              type="text"
              value={projectId}
              onChange={(e) => handleProjectIdChange(e.target.value)}
              placeholder="P0001"
              className={styles.projectInput}
              disabled={isRunning}
            />
          </div>
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

        {/* Parameter Configuration - Optimization Type 선택 후에만 표시 */}
        {optimizationType && (
          <OptimizationParameterSection
            optimizationType={optimizationType}
            optimizationParams={optimizationParams}
            onParamChange={handleParamChange}
            onReset={resetOptimization}
            isRunning={isRunning}
          />
        )}
        
        {/* Execution - Optimization Type 선택 후에만 표시 */}
        {optimizationType && (
          <OptimizationExecution
            isRunning={isRunning}
            progress={progress}
            status={status}
            results={results}
            onRunOptimization={handleRunOptimization}
            optimizationType={optimizationType}
            optimizationParams={optimizationParams}
          />
        )}
      </div>
    </div>
  );
};

export default OptimizationPage;
