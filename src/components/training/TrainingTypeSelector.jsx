import React from 'react';
import TabNavigation from '../common/TabNavigation.jsx';
import { TRAINING_TABS } from '../../domain/training/trainingTypes.js';

const TrainingTypeSelector = ({ trainingType, onTrainingTypeChange }) => {
  return (
    <TabNavigation
      tabs={TRAINING_TABS}
      activeTab={trainingType}
      onTabChange={onTrainingTypeChange}
    />
  );
};

export default TrainingTypeSelector; 