import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const defaultModels = [
  { value: 'artifacts/P0001/training/T0001/yolov8n.pt', label: 'YOLOv8n (Trained)' },
  { value: 'artifacts/P0001/training/T0002/yolov8s.pt', label: 'YOLOv8s (Trained)' },
  { value: 'artifacts/P0001/training/T0003/yolov8m.pt', label: 'YOLOv8m (Trained)' },
  { value: 'artifacts/P0001/training/T0004/yolov8l.pt', label: 'YOLOv8l (Trained)' },
  { value: 'artifacts/P0001/training/T0005/yolov8x.pt', label: 'YOLOv8x (Trained)' },
];

export default function ModelSelector({ 
  selectedModel, 
  onModelChange, 
  disabled, 
  models = defaultModels
}) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Model</label>
      <select
        className={dsStyles.select}
        value={selectedModel || ''}
        onChange={e => onModelChange(e.target.value)}
        disabled={disabled}
      >
        <option value="">Select model</option>
        {models.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
} 