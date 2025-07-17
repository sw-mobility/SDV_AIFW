import React from 'react';
import { HelpCircle, Plus, Minus } from 'lucide-react';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import styles from './ParameterEditor.module.css';
import { validateParam, normalizeParamValue, getDecimalPlaces } from '../../services/trainingService.js';

const ParameterEditor = ({ 
  currentParam, 
  algoParams, 
  onParamChange, 
  paramErrors, 
  isTraining = false 
}) => {
  if (!currentParam) {
    return (
      <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
        <span style={{ color: '#aaa', fontSize: 15 }}>왼쪽에서 파라미터를 선택하세요.</span>
      </div>
    );
  }

  const handleParamChange = (key, value, param) => {
    const newValue = normalizeParamValue(value, param);
    onParamChange(key, newValue, param);
  };

  const inputSizeStyle = { 
    width: 150, 
    height: 38, 
    fontSize: 15, 
    fontWeight: 500, 
    borderRadius: 6, 
    background: '#fff', 
    minWidth: 0, 
    maxWidth: '100%' 
  };

  return (
    <div className={styles.paramCard + ' ' + styles.paramCardActive}>
      <div className={styles.paramRowHeader}>
        <span className={styles.paramLabel}>{currentParam.label}</span>
        <Tooltip 
          title={`기본값: ${currentParam.default}${currentParam.min !== undefined ? `, 범위: ${currentParam.min}~${currentParam.max}` : ''}`.trim()} 
          placement="right"
        >
          <HelpCircle size={18} color="#888" style={{ marginLeft: 6, verticalAlign: 'middle' }} />
        </Tooltip>
      </div>
      
      {/* Input field by type */}
      {currentParam.type === 'number' ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Slider
            min={currentParam.min}
            max={currentParam.max}
            step={currentParam.step}
            value={algoParams[currentParam.key] ?? currentParam.default}
            onChange={(_, v) => handleParamChange(currentParam.key, v, currentParam)}
            sx={{ width: 180, color: '#4f8cff' }}
          />
          <input
            type="number"
            value={algoParams[currentParam.key] ?? currentParam.default}
            min={currentParam.min}
            max={currentParam.max}
            step={currentParam.step}
            onChange={e => handleParamChange(currentParam.key, Number(e.target.value), currentParam)}
            className={styles.paramInput}
            style={{ width: 80, marginLeft: 8 }}
          />
        </div>
      ) : currentParam.type === 'select' ? (
        <select
          className={styles.paramInput}
          value={algoParams[currentParam.key] ?? currentParam.default}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
        >
          {currentParam.options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      ) : currentParam.type === 'checkbox' ? (
        <div className={styles.switchContainer}>
          <Switch
            checked={algoParams[currentParam.key] ?? currentParam.default}
            onChange={e => handleParamChange(currentParam.key, e.target.checked, currentParam)}
            color="primary"
            size="medium"
          />
        </div>
      ) : (
        <input
          type="text"
          className={styles.paramInput}
          value={algoParams[currentParam.key] ?? currentParam.default}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
        />
      )}
      
      {currentParam.desc && <div className={styles.paramDesc}>{currentParam.desc}</div>}
    </div>
  );
};

export default ParameterEditor; 