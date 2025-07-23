import React from 'react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import { Info } from 'lucide-react';
import Button from '../common/Button.jsx';
import styles from '../training/ParameterEditor.module.css';

const defaultOptionDefs = [
  { key: 'quantization', label: 'Quantization', type: 'checkbox', tooltip: 'Enable quantization to reduce model size and improve inference speed.' },
  { key: 'batchSize', label: 'Batch Size', type: 'number', min: 1, max: 256, step: 1, default: 32 },
  { key: 'optLevel', label: 'Optimization Level', type: 'select', options: ['O1', 'O2', 'O3'] },
];

const OptionEditor = ({ options = {}, onChange, onRun, isRunning }) => (
  <div
    className={styles.paramCard}
    style={{
      margin: '0 0 32px 0',
      width: '100%',
      maxWidth: '100%',
      background: '#fff',
      border: '1.5px solid #e2e8f0',
      boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
      borderRadius: 16,
      padding: '32px 36px 28px 36px',
      boxSizing: 'border-box',
      position: 'relative',
      minHeight: 220,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'flex-start',
    }}
  >
    <div className={styles.paramRowHeader} style={{ fontSize: 18, fontWeight: 700, marginBottom: 24 }}>
      Options
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, alignItems: 'center', marginBottom: 24 }}>
      {/* Quantization */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <label className={styles.paramLabel} style={{ minWidth: 110, marginBottom: 0 }}>
          Quantization
          <Tooltip title={defaultOptionDefs[0].tooltip} placement="right">
            <span style={{ display: 'inline-flex', alignItems: 'center', marginLeft: 4, verticalAlign: 'middle', cursor: 'pointer' }}>
              <Info size={17} color="#94a3b8" />
            </span>
          </Tooltip>
        </label>
        <input
          type="checkbox"
          checked={!!options.quantization}
          onChange={e => onChange({ ...options, quantization: e.target.checked })}
          style={{ marginLeft: 2 }}
        />
      </div>
      {/* Optimization Level */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <label className={styles.paramLabel} style={{ minWidth: 110, marginBottom: 0 }}>Optimization Level</label>
        <select
          className={styles.paramInput}
          value={options.optLevel || ''}
          onChange={e => onChange({ ...options, optLevel: e.target.value })}
          style={{ width: 120 }}
        >
          <option value='' disabled>Select</option>
          {defaultOptionDefs[2].options.map(o => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      </div>
      {/* Batch Size (full row) */}
      <div style={{ gridColumn: '1 / span 2', display: 'flex', alignItems: 'center', gap: 18, marginTop: 8 }}>
        <label className={styles.paramLabel} style={{ minWidth: 110, marginBottom: 0 }}>Batch Size</label>
        <Slider
          min={defaultOptionDefs[1].min}
          max={defaultOptionDefs[1].max}
          step={defaultOptionDefs[1].step}
          value={options.batchSize ?? defaultOptionDefs[1].default}
          onChange={(_, v) => onChange({ ...options, batchSize: v })}
          sx={{ width: 180, color: '#4f8cff' }}
        />
        <input
          type="number"
          className={styles.paramInput}
          value={options.batchSize ?? defaultOptionDefs[1].default}
          min={defaultOptionDefs[1].min}
          max={defaultOptionDefs[1].max}
          step={defaultOptionDefs[1].step}
          onChange={e => onChange({ ...options, batchSize: Number(e.target.value) })}
          style={{ width: 70, marginLeft: 8 }}
        />
      </div>
    </div>
    <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginTop: 'auto' }}>
      <Button
        size="medium"
        variant="primary"
        onClick={onRun}
        disabled={isRunning}
        style={{ minWidth: 160, fontWeight: 600, fontSize: 15 }}
      >
        {isRunning ? 'Running...' : 'Run Optimization'}
      </Button>
    </div>
  </div>
);

export default OptionEditor; 