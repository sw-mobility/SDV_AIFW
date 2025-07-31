import React from 'react';
import Selector from '../../ui/Selector.jsx';

const testDatasets = [
  { value: 'test1', label: 'Test Dataset 1' },
  { value: 'test2', label: 'Test Dataset 2' },
  { value: 'test3', label: 'Test Dataset 3' },
];

export default function TestDatasetSelector({ value, onChange, disabled }) {
  return (
    <Selector
      label="Test Dataset"
      options={testDatasets}
      value={value}
      onChange={onChange}
      disabled={disabled}
      placeholder="Select Test Dataset"
    />
  );
} 