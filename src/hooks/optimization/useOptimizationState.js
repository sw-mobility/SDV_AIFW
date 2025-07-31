import { useState, useCallback } from 'react';
import { useProgress } from '../common/useProgress.js';

const useOptimizationState = () => {
  const [targetBoard, setTargetBoard] = useState('');
  const [model, setModel] = useState('');
  const [testDataset, setTestDataset] = useState('');
  const [options, setOptions] = useState({});
  
  const progress = useProgress();

  const runOptimization = useCallback(() => {
    if (!targetBoard || !model || !testDataset) {
      progress.addLog("Please select all required fields.");
      return;
    }

    progress.start();
    progress.addLog("Optimization started...");

    let pct = 0;
    const interval = setInterval(() => {
      pct += 10;
      progress.updateProgress(pct);
      progress.addLog(`Progress: ${pct}%`);
      
      if (pct >= 100) {
        clearInterval(interval);
        progress.complete();
        progress.addLog('Optimization completed!');
      }
    }, 400);
  }, [targetBoard, model, testDataset, progress]);

  const resetOptimization = useCallback(() => {
    setTargetBoard('');
    setModel('');
    setTestDataset('');
    setOptions({});
    progress.reset();
  }, [progress]);

  return {
    targetBoard, 
    setTargetBoard,
    model, 
    setModel,
    testDataset, 
    setTestDataset,
    options, 
    setOptions,
    isRunning: progress.isRunning,
    progress: progress.progress,
    logs: progress.logs,
    status: progress.status,
    runOptimization,
    resetOptimization,
  };
};

export default useOptimizationState; 