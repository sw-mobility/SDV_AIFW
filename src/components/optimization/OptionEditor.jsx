import React from 'react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Checkbox from '@mui/material/Checkbox';
import { Info } from 'lucide-react';
import Button from '../common/Button.jsx';
import styles from './OptionEditor.module.css';

const optionDefs = [
  { key: 'quantization', label: 'Quantization', type: 'checkbox', tooltip: 'Enable quantization to reduce model size and improve inference speed.' },
  { key: 'batchSize', label: 'Batch Size', type: 'number', min: 1, max: 256, step: 1, default: 32 },
  { key: 'optLevel', label: 'Optimization Level', type: 'select', options: ['O1', 'O2', 'O3'] },
];

export default function OptionEditor({ options = {}, onChange, onRun, isRunning }) {
  return (
    <div className={styles.paramCard}>
      <div className={styles.paramRowHeader}>Options</div>
      <div className={styles.optionsGrid}>
        {/* Quantization */}
        <div className={styles.optionRow}>
          <label className={styles.paramLabel}>
            Quantization
          </label>
          <Checkbox
            checked={!!options.quantization}
            onChange={e => onChange({ ...options, quantization: e.target.checked })}
            disabled={isRunning}
            sx={{ '& .MuiSvgIcon-root': { fontSize: 24 }, padding: '4px' }}
          />
        </div>
        {/* Optimization Level */}
        <div className={styles.optionRow}>
          <label className={styles.paramLabel}>Optimization Level</label>
          <select
            className={styles.paramInput}
            value={options.optLevel || ''}
            onChange={e => onChange({ ...options, optLevel: e.target.value })}
            disabled={isRunning}
          >
            <option value='' disabled>Select</option>
            {optionDefs[2].options.map(o => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        {/* Batch Size */}
        <div className={styles.optionRow} style={{ gridColumn: '1 / span 2' }}>
          <label className={styles.paramLabel}>Batch Size</label>
          <Slider
            min={optionDefs[1].min}
            max={optionDefs[1].max}
            step={optionDefs[1].step}
            value={options.batchSize ?? optionDefs[1].default}
            onChange={(_, v) => onChange({ ...options, batchSize: v })}
            sx={{ width: 180, color: '#4f8cff' }}
            disabled={isRunning}
          />
          <input
            type="number"
            className={styles.paramInput}
            value={options.batchSize ?? optionDefs[1].default}
            min={optionDefs[1].min}
            max={optionDefs[1].max}
            step={optionDefs[1].step}
            onChange={e => onChange({ ...options, batchSize: Number(e.target.value) })}
            disabled={isRunning}
          />
        </div>
      </div>
      <div className={styles.buttonRow}>
        <Button
          size="medium"
          variant="primary"
          onClick={onRun}
          disabled={isRunning}
        >
          {isRunning ? 'Running...' : 'Run Optimization'}
        </Button>
      </div>
    </div>
  );
} 