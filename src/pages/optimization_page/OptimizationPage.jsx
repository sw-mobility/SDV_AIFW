import React from 'react';
import SectionTitle from '../../components/common/SectionTitle.jsx';
import styles from './OptimizationPage.module.css';
// Placeholder imports for new components/hooks
import TargetBoardSelector from '../../components/optimization/TargetBoardSelector.jsx';
import ModelSelector from '../../components/optimization/ModelSelector.jsx';
import TestDatasetSelector from '../../components/optimization/TestDatasetSelector.jsx';
import OptionEditor from '../../components/optimization/OptionEditor.jsx';
import OptimizationExecution from '../../components/optimization/OptimizationExecution.jsx';
import useOptimizationState from '../../hooks/useOptimizationState.js';

const OptimizationPage = () => {
  const {
    targetBoard, setTargetBoard,
    model, setModel,
    testDataset, setTestDataset,
    options, setOptions,
    isRunning, progress, logs, status,
    runOptimization,
  } = useOptimizationState();

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Optimization</h1>
        <p className={styles.pageDescription}>
          Configure your optimization settings and run model optimization with your selected board, model, and dataset.
        </p>
      </div>
      <div className={styles.sectionWrap}>
        <div className={styles.leftPanel}>
          <TargetBoardSelector value={targetBoard} onChange={setTargetBoard} />
          <ModelSelector value={model} onChange={setModel} />
          <TestDatasetSelector value={testDataset} onChange={setTestDataset} />
        </div>
        <div className={styles.rightPanel}>
          <OptionEditor options={options} onChange={setOptions} />
          <OptimizationExecution
            isRunning={isRunning}
            progress={progress}
            logs={logs}
            status={status}
            onRun={runOptimization}
          />
        </div>
      </div>
    </div>
  );
};

export default OptimizationPage;
