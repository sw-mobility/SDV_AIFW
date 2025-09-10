// Training Type Domain Model
export const TRAINING_TYPES = {
  STANDARD: 'standard',
  CONTINUAL: 'continual'
};

export const TRAINING_TYPE_LABELS = {
  [TRAINING_TYPES.STANDARD]: 'Standard Training',
  [TRAINING_TYPES.CONTINUAL]: 'Continual Training'
};

export const TRAINING_TABS = [
  { value: TRAINING_TYPES.STANDARD, label: TRAINING_TYPE_LABELS[TRAINING_TYPES.STANDARD] },
  { value: TRAINING_TYPES.CONTINUAL, label: TRAINING_TYPE_LABELS[TRAINING_TYPES.CONTINUAL] }
];

export const TRAINING_TYPE_CONFIG = {
  [TRAINING_TYPES.STANDARD]: {
    label: 'Standard Training',
    description: 'Train a new model from scratch',
    requiresSnapshot: false,
    requiresBaseModel: false
  },
  [TRAINING_TYPES.CONTINUAL]: {
    label: 'Continual Training',
    description: 'Continue training from an existing model',
    requiresSnapshot: true,
    requiresBaseModel: true
  }
};

export const getTrainingTypeConfig = (type) => {
  return TRAINING_TYPE_CONFIG[type] || TRAINING_TYPE_CONFIG[TRAINING_TYPES.STANDARD];
};

export const validateTrainingType = (type) => {
  return Object.values(TRAINING_TYPES).includes(type);
}; 