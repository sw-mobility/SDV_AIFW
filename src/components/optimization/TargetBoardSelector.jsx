import React from 'react';
import Selector from '../common/Selector.jsx';

const boards = [
  { value: 'board1', label: 'Board 1' },
  { value: 'board2', label: 'Board 2' },
  { value: 'board3', label: 'Board 3' },
];

export default function TargetBoardSelector({ value, onChange, disabled }) {
  return (
    <Selector
      label="Target Board"
      options={boards}
      value={value}
      onChange={onChange}
      disabled={disabled}
      placeholder="Select Board"
    />
  );
} 