import React from 'react';
import { HelpCircle } from 'lucide-react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import styles from './OptimizationParameterEditor.module.css';
import { normalizeOptimizationParamValue } from '../../../domain/optimization/optimizationParameters.js';

/**
 * Optimization 파라미터 입력 및 검증 담당
 * Training/Validation 페이지의 ParameterEditor와 동일한 구조
 */
const OptimizationParameterEditor = ({ 
  currentParam, 
  optimizationParams, 
  onParamChange,
  isRunning = false
}) => {
  if (!currentParam) {
    return null;
  }

  const handleParamChange = (key, value, param) => {
    const newValue = normalizeOptimizationParamValue(value, param);
    onParamChange(key, newValue, param);
  };

  // 슬라이더용 step 값 계산
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

  // 현재 값 가져오기 (기본값 포함)
  const getCurrentValue = () => {
    return optimizationParams[currentParam.key] !== undefined 
      ? optimizationParams[currentParam.key] 
      : currentParam.default;
  };

  // Array 타입을 위한 특별한 렌더링
  const renderArrayInput = () => {
    const value = getCurrentValue();
    const [width, height] = Array.isArray(value) ? value : [640, 640];
    
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ fontSize: '12px', color: '#666' }}>Width</label>
          <input
            type="number"
            value={width}
            min={224}
            max={1024}
            step={32}
            onChange={e => {
              const newWidth = Number(e.target.value);
              handleParamChange(currentParam.key, [newWidth, height], currentParam);
            }}
            className={styles.paramInput}
            style={{ width: 80 }}
            disabled={isRunning}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ fontSize: '12px', color: '#666' }}>Height</label>
          <input
            type="number"
            value={height}
            min={224}
            max={1024}
            step={32}
            onChange={e => {
              const newHeight = Number(e.target.value);
              handleParamChange(currentParam.key, [width, newHeight], currentParam);
            }}
            className={styles.paramInput}
            style={{ width: 80 }}
            disabled={isRunning}
          />
        </div>
      </div>
    );
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
      {currentParam.type === 'number' ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Slider
            min={currentParam.min}
            max={currentParam.max}
            step={getSliderStep(currentParam)}
            value={getCurrentValue()}
            onChange={(_, v) => handleParamChange(currentParam.key, v, currentParam)}
            disabled={isRunning}
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
            disabled={isRunning}
          />
        </div>
      ) : currentParam.type === 'select' ? (
        <select
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
          disabled={isRunning}
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
            color="primary"
            size="medium"
            disabled={isRunning}
          />
          <span style={{ marginLeft: 8, fontSize: '14px', color: '#666' }}>
            {getCurrentValue() ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      ) : currentParam.type === 'array' ? (
        renderArrayInput()
      ) : (
        <input
          type="text"
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
          placeholder={currentParam.placeholder || ''}
          disabled={isRunning}
        />
      )}
      
      {currentParam.desc && <div className={styles.paramDesc}>{currentParam.desc}</div>}
    </div>
  );
};

export default OptimizationParameterEditor;




