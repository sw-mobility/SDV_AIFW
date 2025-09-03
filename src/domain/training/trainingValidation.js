import { TRAINING_TYPES } from './trainingTypes.js';

export class TrainingValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'TrainingValidationError';
    this.field = field;
  }
}

export const validateTrainingExecution = (trainingConfig) => {
  const errors = [];
  const { trainingType, selectedDataset, selectedSnapshot, algorithm, modelType, customModel } = trainingConfig;

  // Dataset validation
  if (!selectedDataset) {
    errors.push(new TrainingValidationError('데이터셋을 선택해주세요.', 'dataset'));
  }

  // Snapshot validation for continual training
  if (trainingType === TRAINING_TYPES.CONTINUAL && !selectedSnapshot) {
    errors.push(new TrainingValidationError('Continual training을 위해서는 기본 스냅샷을 선택해야 합니다.', 'snapshot'));
  }

  // Algorithm validation - modelType에 따라 다르게 처리
  if (modelType === 'pretrained' && !algorithm) {
    errors.push(new TrainingValidationError('알고리즘을 선택해주세요.', 'algorithm'));
  } else if (modelType === 'custom' && !customModel) {
    errors.push(new TrainingValidationError('커스텀 모델을 선택해주세요.', 'customModel'));
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

export const validateParameter = (param, value) => {
  let error = '';
  
  if (param.type === 'number') {
    if (typeof value !== 'number' || isNaN(value)) {
      error = '숫자를 입력하세요.';
    } else if (param.min !== undefined && value < param.min) {
      error = `${param.label}은(는) 최소 ${param.min} 이상이어야 합니다.`;
    } else if (param.max !== undefined && value > param.max) {
      error = `${param.label}은(는) 최대 ${param.max} 이하여야 합니다.`;
    }
  } else if (param.type === 'text') {
    if (param.required && (!value || value === '')) {
      error = `${param.label}을(를) 입력하세요.`;
    }
  } else if (param.type === 'yaml_editor') {
    if (param.required && (!value || value === '')) {
      error = `${param.label}을(를) 입력하세요.`;
    }
  }
  
  return { isValid: error === '', error };
}; 