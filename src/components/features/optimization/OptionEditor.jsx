import React from 'react';
import Checkbox from '@mui/material/Checkbox';
import Button from '../../ui/atoms/Button.jsx';
import styles from './OptionEditor.module.css';

export default function OptionEditor({
  options = {},
  onChange,
  onRun,
  isRunning,
  parameterDefs = [],
  targetBoard,
  model
}) {
  return (
    <div className={styles.paramCard}>
      <div className={styles.paramRowHeader}>Parameters</div>
      <div style={{marginBottom: 12, color: '#64748b', fontSize: 14}}>
        선택된 Target Board: <b>{targetBoard || '-'}</b> / Model: <b>{model || '-'}</b>
      </div>
      <div className={styles.optionsGrid}>
        {parameterDefs.length === 0 && (
          <div style={{gridColumn: '1 / span 2', color: '#b91c1c'}}>이 조합에 맞는 파라미터가 없습니다.</div>
        )}
        {parameterDefs.map(def => {
          if (def.type === 'checkbox') {
            return (
              <div className={styles.optionRow} key={def.key}>
                <label className={styles.paramLabel}>{def.label}</label>
                <Checkbox
                  checked={!!options[def.key]}
                  onChange={e => onChange({ ...options, [def.key]: e.target.checked })}
                  disabled={isRunning}
                  sx={{ '& .MuiSvgIcon-root': { fontSize: 24 }, padding: '4px' }}
                />
              </div>
            );
          }
          if (def.type === 'select') {
            return (
              <div className={styles.optionRow} key={def.key}>
                <label className={styles.paramLabel}>{def.label}</label>
                <select
                  className={styles.paramInput}
                  value={options[def.key] || ''}
                  onChange={e => onChange({ ...options, [def.key]: e.target.value })}
                  disabled={isRunning}
                >
                  <option value='' disabled>Select</option>
                  {def.options.map(o => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
            );
          }
          if (def.type === 'number') {
            return (
              <div className={styles.optionRow} key={def.key}>
                <label className={styles.paramLabel}>{def.label}</label>
                <input
                  type="number"
                  className={styles.paramInput}
                  value={options[def.key] ?? def.default ?? ''}
                  min={def.min}
                  max={def.max}
                  step={def.step}
                  onChange={e => onChange({ ...options, [def.key]: Number(e.target.value) })}
                  disabled={isRunning}
                />
              </div>
            );
          }
          return null;
        })}
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