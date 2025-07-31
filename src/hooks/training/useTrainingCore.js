import { useState, useCallback } from 'react';
import { TRAINING_TYPES } from '../../domain/training/trainingTypes.js';

export const useTrainingCore = () => {
  const [trainingType, setTrainingType] = useState(TRAINING_TYPES.STANDARD);
  const [algorithm, setAlgorithm] = useState('YOLO');
  const [algoParams, setAlgoParams] = useState({});
  const [paramErrors, setParamErrors] = useState({});

  const resetTraining = useCallback(() => {
    setTrainingType(TRAINING_TYPES.STANDARD);
    setAlgorithm('YOLO');
    setAlgoParams({});
    setParamErrors({});
  }, []);

  const updateAlgorithm = useCallback((newAlgorithm) => {
    setAlgorithm(newAlgorithm);
    setAlgoParams({});
    setParamErrors({});
  }, []);

  const updateParam = useCallback((key, value) => {
    setAlgoParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const updateParamError = useCallback((key, error) => {
    setParamErrors(prev => ({ ...prev, [key]: error }));
  }, []);

  return {
    trainingType,
    setTrainingType,
    algorithm,
    setAlgorithm: updateAlgorithm,
    algoParams,
    setAlgoParams,
    paramErrors,
    setParamErrors,
    updateParam,
    updateParamError,
    resetTraining
  };
}; 