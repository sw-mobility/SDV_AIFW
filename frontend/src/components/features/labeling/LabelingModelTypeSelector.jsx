import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const MODEL_TYPES = [
  { 
    value: 'detection', 
    label: 'Detection',
    status: 'available'
  },
  { 
    value: 'segmentation', 
    label: 'Segmentation',
    status: 'coming_soon'
  },
  { 
    value: 'pose', 
    label: 'Pose Estimation',
    status: 'coming_soon'
  },
  { 
    value: 'obb', 
    label: 'Oriented Bounding Boxes',
    status: 'coming_soon'
  },
  { 
    value: 'cls', 
    label: 'Classification',
    status: 'coming_soon'
  }
];

export default function LabelingModelTypeSelector({ 
  modelType, 
  onModelTypeChange, 
  disabled = false
}) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Model Type</label>
      <select
        className={dsStyles.select}
        value={modelType || 'detection'}
        onChange={e => onModelTypeChange(e.target.value)}
        disabled={disabled}
      >
        {MODEL_TYPES.map(type => (
          <option 
            key={type.value} 
            value={type.value}
            disabled={type.status === 'coming_soon'}
          >
            {type.label} {type.status === 'coming_soon' ? '(Coming Soon)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
