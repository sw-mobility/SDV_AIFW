import React from 'react';
import dsStyles from '../training/DatasetSelector.module.css';

const LABELING_FORMATS = [
  { 
    value: 'yolo', 
    label: 'YOLO',
    status: 'available'
  },
  { 
    value: 'coco', 
    label: 'COCO',
    status: 'coming_soon'
  },
  { 
    value: 'pascal_voc', 
    label: 'Pascal VOC',
    status: 'coming_soon'
  }
];

export default function LabelingFormatSelector({ 
  labelingFormat, 
  onLabelingFormatChange, 
  disabled = false
}) {
  return (
    <div className={dsStyles.selectorBox}>
      <label className={dsStyles.paramLabel} style={{marginBottom: 4}}>Format</label>
      <select
        className={dsStyles.select}
        value={labelingFormat || 'yolo'}
        onChange={e => onLabelingFormatChange(e.target.value)}
        disabled={disabled}
      >
        {LABELING_FORMATS.map(format => (
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
