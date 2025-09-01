import React from 'react';
import styles from './OptimizationPage.module.css';
import OptimizationTypeSelector from '../../components/features/optimization/OptimizationTypeSelector.jsx';
import OptimizationParameterSection from '../../components/features/optimization/OptimizationParameterSection.jsx';
import OptimizationExecution from '../../components/features/optimization/OptimizationExecution.jsx';
import { useOptimizationState } from '../../hooks';

const OptimizationPage = () => {
  const {
    // Core state
    optimizationType,
    optimizationParams,
    
    // Execution state
    isRunning,
    progress,
    status,
    results,
    
    // Event handlers
    handleRunOptimization,
    handleOptimizationTypeChange,
    handleParamChange,
    resetOptimization
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
          onReset={resetOptimization}
          isRunning={isRunning}
        />
        
        {/* Execution */}
        <OptimizationExecution
          isRunning={isRunning}
          progress={progress}
          status={status}
          results={results}
          onRunOptimization={handleRunOptimization}
          optimizationType={optimizationType}
          optimizationParams={optimizationParams}
        />
      </div>
    </div>
  );
};

export default OptimizationPage;
