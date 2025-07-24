import React from 'react';
import Selector from '../common/Selector.jsx';

const models = [
  { value: 'modelA', label: 'Model A' },
  { value: 'modelB', label: 'Model B' },
  { value: 'modelC', label: 'Model C' },
];

export default function ModelSelector({ value, onChange, disabled }) {
  return (
    <Selector
      label="Model"
      options={models}
      value={value}
      onChange={onChange}
      disabled={disabled}
      placeholder="Select Model"
    />
  );
} 