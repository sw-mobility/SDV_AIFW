import React from 'react';
import styles from './OptimizationPage.module.css';
import ModelSelector from '../../components/features/optimization/ModelSelector.jsx';
import OptimizationTypeSelector from '../../components/features/optimization/OptimizationTypeSelector.jsx';
import OptimizationParameterSection from '../../components/features/optimization/OptimizationParameterSection.jsx';
import OptimizationExecution from '../../components/features/optimization/OptimizationExecution.jsx';
import { useOptimizationState } from '../../hooks';

const OptimizationPage = () => {
  const {
    // Core state
    selectedModel,
    optimizationType,
    optimizationParams,
    
    // Execution state
    isRunning,
    progress,
    status,
    logs,
    results,
    
    // Event handlers
    handleRunOptimization,
    handleModelChange,
    handleOptimizationTypeChange,
    handleParamChange,
    handleReset
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

        {/* Selectors */}
        <div className={styles.selectorGroup}>
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            disabled={isRunning}
          />
          <OptimizationTypeSelector
            optimizationType={optimizationType}
            onOptimizationTypeChange={handleOptimizationTypeChange}
            disabled={isRunning}
          />
        </div>

        {/* Parameter Configuration */}
        <OptimizationParameterSection
          optimizationType={optimizationType}
          optimizationParams={optimizationParams}
          onParamChange={handleParamChange}
          onReset={handleReset}
          isRunning={isRunning}
        />
        
        {/* Execution */}
        <OptimizationExecution
          isRunning={isRunning}
          progress={progress}
          status={status}
          logs={logs}
          results={results}
          onRunOptimization={handleRunOptimization}
          selectedModel={selectedModel}
          optimizationType={optimizationType}
        />
      </div>
    </div>
  );
};

export default OptimizationPage;
