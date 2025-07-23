import { useState } from 'react';

export default function useOptimizationState() {
  const [targetBoard, setTargetBoard] = useState('');
  const [model, setModel] = useState('');
  const [testDataset, setTestDataset] = useState('');
  const [options, setOptions] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');

  // Mock optimization run
  const runOptimization = () => {
    if (!targetBoard || !model || !testDataset) {
      setLogs(["Please select all required fields."]);
      return;
    }
    setIsRunning(true);
    setStatus('running');
    setLogs(["Optimization started..."]);
    setProgress(0);
    let pct = 0;
    const interval = setInterval(() => {
      pct += 10;
      setProgress(pct);
      setLogs(l => [...l, `Progress: ${pct}%`]);
      if (pct >= 100) {
        clearInterval(interval);
        setIsRunning(false);
        setStatus('success');
        setLogs(l => [...l, 'Optimization completed!']);
      }
    }, 400);
  };

  return {
    targetBoard, setTargetBoard,
    model, setModel,
    testDataset, setTestDataset,
    options, setOptions,
    isRunning, progress, logs, status,
    runOptimization,
  };
} 