import { useState, useCallback} from 'react';

export const useProgress = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState([]);

  const start = useCallback(() => {
    setIsRunning(true);
    setStatus('running');
    setProgress(0);
    setLogs([]);
  }, []);

  const stop = useCallback(() => {
    setIsRunning(false);
    setStatus('stopped');
  }, []);

  const complete = useCallback(() => {
    setIsRunning(false);
    setStatus('completed');
    setProgress(100);
  }, []);

  const addLog = useCallback((message) => {
    setLogs(prev => [...prev, message]);
  }, []);

  const updateProgress = useCallback((newProgress) => {
    setProgress(Math.min(100, Math.max(0, newProgress)));
  }, []);

  const reset = useCallback(() => {
    setIsRunning(false);
    setProgress(0);
    setStatus('idle');
    setLogs([]);
  }, []);

  return {
    isRunning,
    progress,
    status,
    logs,
    start,
    stop,
    complete,
    addLog,
    updateProgress,
    reset
  };
}; 