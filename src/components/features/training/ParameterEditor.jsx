import React from 'react';
import { HelpCircle } from 'lucide-react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import styles from './ParameterEditor.module.css';
import { normalizeParamValue} from '../../../domain/training/parameterGroups.js';
import COCOClassesEditor from './COCOClassesEditor.jsx';

/**
 * 파라미터 입력 및 검증 담당
 *
 * @param currentParam
 * @param algoParams
 * @param onParamChange
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const ParameterEditor = ({ 
  currentParam, 
  algoParams, 
  onParamChange,
}) => {
  if (!currentParam) {
    return null; // 빈 상태는 ParameterSection에서 처리
  }

  const handleParamChange = (key, value, param) => {
    const newValue = normalizeParamValue(value, param);
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
    return algoParams[currentParam.key] !== undefined 
      ? algoParams[currentParam.key] 
      : currentParam.default;
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
            step={getSliderStep(currentParam)}
            value={getCurrentValue()}
            onChange={(_, v) => handleParamChange(currentParam.key, v, currentParam)}
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
          />
        </div>
      ) : currentParam.type === 'select' ? (
        <select
          className={styles.paramInput}
          value={getCurrentValue()}
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
            checked={getCurrentValue()}
            onChange={e => handleParamChange(currentParam.key, e.target.checked, currentParam)}
            color="primary"
            size="medium"
          />
          <span style={{ marginLeft: 8, fontSize: '14px', color: '#666' }}>
            {getCurrentValue() ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      ) : currentParam.type === 'yaml_editor' ? (
        <div className={styles.yamlEditorContainer}>
          <COCOClassesEditor
            value={getCurrentValue()}
            onChange={(value) => handleParamChange(currentParam.key, value, currentParam)}
          />
        </div>
      ) : (
        <input
          type="text"
          className={styles.paramInput}
          value={getCurrentValue()}
          onChange={e => handleParamChange(currentParam.key, e.target.value, currentParam)}
          style={{ width: 180 }}
          placeholder={currentParam.placeholder || ''}
        />
      )}
      
      {currentParam.desc && <div className={styles.paramDesc}>{currentParam.desc}</div>}
    </div>
  );
};

export default ParameterEditor; 