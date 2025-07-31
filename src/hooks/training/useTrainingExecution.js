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

    try {
      const result = await executeTraining(trainingConfig);
      
      if (result.success) {
        progress.addLog(result.message);
      }
    } catch (error) {
      progress.addLog(`Error: ${error.message}`);
      progress.stop();
    }
  }, [trainingConfig, progress]);

  return {
    ...progress,
    runTraining
  };
}; 