import React from 'react';
import dsStyles from './DatasetSelector.module.css';

const TRAINING_FORMATS = [
  { 
    value: 'yolo', 
    label: 'YOLO',
    status: 'available'
  },
  { 
    value: 'rcnn', 
    label: 'R-CNN',
    status: 'coming_soon'
  },
  { 
    value: 'ssd', 
    label: 'SSD',
    status: 'coming_soon'
  },
  { 
    value: 'retinanet', 
    label: 'RetinaNet',
    status: 'coming_soon'
  }
];

export default function TrainingFormatSelector({ 
  trainingFormat, 
  onTrainingFormatChange, 
  disabled = false
}) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Framework</label>
      <select
        className={dsStyles.select}
        value={trainingFormat || 'yolo'}
        onChange={e => onTrainingFormatChange(e.target.value)}
        disabled={disabled}
      >
        {TRAINING_FORMATS.map(format => (
          <option 
            key={format.value} 
            value={format.value}
            disabled={format.status === 'coming_soon'}
          >
            {format.label} {format.status === 'coming_soon' ? '(Coming Soon)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
