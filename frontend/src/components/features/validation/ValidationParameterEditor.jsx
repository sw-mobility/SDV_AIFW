import React from 'react';
import { HelpCircle } from 'lucide-react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import TrainingIdSelector from '../training/TrainingIdSelector.jsx';
import styles from './ValidationParameterEditor.module.css';
import { normalizeValidationParamValue } from '../../../domain/validation/validationParameters.js';

/**
 * Validation 파라미터 입력 및 검증 담당
 * Training 페이지의 ParameterEditor와 정확히 동일한 구조
 *
 * @param currentParam
 * @param validationParams
 * @param onParamChange
 * @param disabled
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ValidationParameterEditor = ({ 
  currentParam, 
  validationParams, 
  onParamChange,
  disabled = false,
  projectId = 'P0001'
}) => {
  if (!currentParam) {
    return null; // 빈 상태는 ValidationParameterSection에서 처리
  }

  const handleParamChange = (key, value, param) => {
    // disabled된 파라미터는 변경 불가
    if (param.disabled) return;
    
    const newValue = normalizeValidationParamValue(value, param);
    onParamChange({ [key]: newValue });
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
    return validationParams[currentParam.key] !== undefined 
      ? validationParams[currentParam.key] 
      : currentParam.default;
  };

  return (
    <div className={`${styles.paramCard} ${styles.paramCardActive} ${currentParam.disabled ? styles.paramCardDisabled : ''}`}>
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
            step={getSliderStep(currentParam)}
            value={getCurrentValue()}
            onChange={(_, v) => handleParamChange(currentParam.key, v, currentParam)}
            disabled={disabled || currentParam.disabled}
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
            disabled={disabled || currentParam.disabled}
            style={{ width: 80, marginLeft: 8 }}
          />
        </div>
      ) : currentParam.type === 'select' ? (
        <select
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          disabled={disabled || currentParam.disabled}
          style={{ width: 180 }}
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
            disabled={disabled || currentParam.disabled}
            color="primary"
            size="medium"
          />
          <span style={{ marginLeft: 8, fontSize: '14px', color: '#666' }}>
            {getCurrentValue() ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      ) : currentParam.key === 'tid' ? (
        <TrainingIdSelector
          selectedTid={getCurrentValue()}
          onTidChange={(value) => handleParamChange(currentParam.key, value, currentParam)}
          projectId={projectId}
          showCompletedOnly={true}
          placeholder="Select Training ID"
          disabled={disabled || currentParam.disabled}
        />
      ) : (
        <input
          type="text"
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          disabled={disabled || currentParam.disabled}
          style={{ width: 180 }}
          placeholder={currentParam.placeholder || ''}
        />
      )}
      
      {currentParam.desc && <div className={styles.paramDesc}>{currentParam.desc}</div>}
    </div>
  );
};

export default ValidationParameterEditor;
