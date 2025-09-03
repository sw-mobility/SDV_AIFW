import React from 'react';
import styles from './OptimizationPage.module.css';
import OptimizationTypeSelector from '../../components/features/optimization/OptimizationTypeSelector.jsx';
import OptimizationModelSelector from '../../components/features/optimization/OptimizationModelSelector.jsx';
import OptimizationParameterSection from '../../components/features/optimization/OptimizationParameterSection.jsx';
import OptimizationExecution from '../../components/features/optimization/OptimizationExecution.jsx';
import OptimizationHistoryList from '../../components/features/optimization/OptimizationHistoryList.jsx';
import { useOptimizationState } from '../../hooks';

const OptimizationPage = () => {
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
    refreshOptimizationHistory
  } = useOptimizationState();

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
          disabled={isRunning}
        />

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
            onRunOptimization={handleRunOptimization}
            optimizationType={optimizationType}
            optimizationParams={optimizationParams}
          />
        )}
      </div>

      {/* Optimization History List - 항상 표시 */}
      <div className={`${styles.container} ${styles.historyContainer}`}>
        <OptimizationHistoryList onRefresh={refreshOptimizationHistory} />
      </div>
    </div>
  );
};

export default OptimizationPage;
