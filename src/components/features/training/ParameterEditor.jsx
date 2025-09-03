import React from 'react';
import { Slider, Tooltip, Switch } from '@mui/material';
import { HelpCircle } from 'lucide-react';
import COCOClassesEditor from './COCOClassesEditor.jsx';
import styles from './ParameterEditor.module.css';

const ParameterEditor = ({ 
  currentParam, 
  algoParams, 
  onParamChange, 
  paramErrors, 
  disabled = false,
  selectedDataset
}) => {
  if (!currentParam) {
    return null;
  }

  const getCurrentValue = () => {
    return algoParams[currentParam.key] !== undefined 
      ? algoParams[currentParam.key] 
      : currentParam.default;
  };

  const handleParamChange = (key, value, param) => {
    onParamChange(key, value, param);
  };

  const getSliderStep = (param) => {
    if (param.step) return param.step;
    
    // step이 정의되지 않은 경우 범위에 따라 적절한 값 설정
    const range = param.max - param.min;
    if (range <= 1) return 0.01;
    if (range <= 10) return 0.1;
    if (range <= 100) return 1;
    if (range <= 1000) return 10;
    return Math.ceil(range / 100);
  };

  return (
    <div className={styles.paramCard + ' ' + styles.paramCardActive}>
      <div className={styles.paramRowHeader}>
        <span className={styles.paramLabel}>{currentParam.label}</span>
        <Tooltip
          title={`기본값: ${JSON.stringify(currentParam.default)}${currentParam.min !== undefined ? `, 범위: ${currentParam.min}~${currentParam.max}` : ''}`.trim()}
          placement="right"
        >
          <HelpCircle size={18} color="#888" style={{ marginLeft: 6, verticalAlign: 'middle' }} />
        </Tooltip>
      </div>

      {/* Input field by type */}
      {currentParam.key === 'coco_classes' ? (
        <COCOClassesEditor
          value={getCurrentValue()}
          onChange={(value) => handleParamChange(currentParam.key, value, currentParam)}
          dataset={selectedDataset}
        />
      ) : currentParam.type === 'number' ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Slider
            min={currentParam.min}
            max={currentParam.max}
            step={getSliderStep(currentParam)}
            value={getCurrentValue()}
            onChange={(_, v) => handleParamChange(currentParam.key, v, currentParam)}
            disabled={disabled}
            sx={{ 
              width: 180, 
              color: '#4f8cff',
              '& .MuiSlider-thumb': {
                width: 16,
                height: 16,
              },
              '& .MuiSlider-track': {
                height: 4,
              },
              '& .MuiSlider-rail': {
                height: 4,
              }
            }}
          />
          <input
            type="number"
            value={getCurrentValue()}
            min={currentParam.min}
            max={currentParam.max}
            step={currentParam.step || getSliderStep(currentParam)}
            onChange={e => handleParamChange(currentParam.key, Number(e.target.value), currentParam)}
            className={styles.paramInput}
            style={{ width: 80, marginLeft: 8 }}
            disabled={disabled}
          />
        </div>
      ) : currentParam.type === 'select' ? (
        <select
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
          disabled={disabled}
        >
          {currentParam.options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      ) : currentParam.type === 'checkbox' ? (
        <div className={styles.switchContainer}>
          <Switch
            checked={getCurrentValue()}
            onChange={e => handleParamChange(currentParam.key, e.target.checked, currentParam)}
            size="medium"
            disabled={disabled}
          />
          <span style={{ marginLeft: 8, fontSize: '14px', color: '#666' }}>
            {getCurrentValue() ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      ) : (
        <input
          type={currentParam.type === 'float' ? 'number' : 'text'}
          value={getCurrentValue()}
          onChange={(e) => handleParamChange(currentParam.key, e.target.value, currentParam)}
          disabled={disabled}
          className={styles.paramInput}
          step={currentParam.type === 'float' ? '0.1' : undefined}
          min={currentParam.min}
          max={currentParam.max}
          style={{ width: 180 }}
          placeholder={currentParam.placeholder || ''}
        />
      )}

      {paramErrors[currentParam.key] && (
        <div className={styles.errorMessage}>
          {paramErrors[currentParam.key]}
        </div>
      )}

      {currentParam.description && (
        <div className={styles.paramDesc}>
          {currentParam.description}
        </div>
      )}
    </div>
  );
};

export default ParameterEditor; 