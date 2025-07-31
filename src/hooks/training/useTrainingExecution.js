import { useCallback } from 'react';
import { validateTrainingExecution } from '../../domain/training/trainingValidation.js';
import { useProgress } from '../common/useProgress.js';
import { executeTraining } from '../../mocks/trainingSnapshots.js';

export const useTrainingExecution = (trainingConfig) => {
  const progress = useProgress();

  const runTraining = useCallback(async () => {
    const validation = validateTrainingExecution(trainingConfig);
    
    if (!validation.isValid) {
      const errorMessages = validation.errors.map(error => error.message);
      alert(errorMessages.join('\n'));
      return;
    }

    progress.start();
    progress.addLog('Training started...');

    let pct = 0;
    const interval = setInterval(() => {
      pct += 10;
      progress.updateProgress(pct);
      progress.addLog(`Progress: ${pct}%`);
      
      if (pct >= 100) {
        clearInterval(interval);
        progress.complete();
        progress.addLog('Training completed!');
      }
    }, 400);
  }, [trainingConfig, progress]);

  return {
    ...progress,
    runTraining
  };
}; 