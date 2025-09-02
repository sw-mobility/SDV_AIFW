import React from 'react';
import { HelpCircle } from 'lucide-react';
import Slider from '@mui/material/Slider';
import { normalizeParamValue} from '../../../domain/training/parameterGroups.js';
import COCOClassesEditor from './COCOClassesEditor.jsx';
import { convertClassesToYaml } from '../../../utils/classUtils.js';
import styles from './ParameterEditor.module.css';

/**
 * 개별 파라미터 편집 컴포넌트
 * 주요 기능: 파라미터 값 편집 및 유효성 검사
 * @param currentParam
 * @param algoParams
 * @param onParamChange
 * @param paramErrors
 * @param isTraining
 * @param selectedDataset
 * @returns {Element}
 * @constructor
 */
const ParameterEditor = ({ 
  currentParam, 
  algoParams, 
  onParamChange, 
  paramErrors, 
  isTraining,
  selectedDataset
}) => {
  if (!currentParam) {
    return null; // 빈 상태는 ParameterSection에서 처리
  }

  const handleParamChange = (key, newValue, param) => {
    onParamChange(key, newValue, param);
  };

  // COCO classes의 기본값을 dataset classes로 설정
  const getDefaultCOCOClasses = () => {
    if (currentParam.key === 'coco_classes' && selectedDataset?.classes) {
      // Dataset의 classes를 YAML 구조로 변환
      return `names:\n${convertClassesToYaml(selectedDataset.classes)}`;
    }
    return currentParam.default;
  };

  // 현재 값 가져오기 (기본값 포함)
  const getCurrentValue = () => {
    if (currentParam.key === 'coco_classes') {
      return algoParams[currentParam.key] !== undefined 
        ? algoParams[currentParam.key] 
        : getDefaultCOCOClasses();
    }
    return algoParams[currentParam.key] !== undefined 
      ? algoParams[currentParam.key] 
      : currentParam.default;
  };

  // 슬라이더용 step 값 계산
  const getSliderStep = (param) => {
    if (param.type === 'float') {
      return param.step || 0.1;
    }
    return param.step || 1;
  };

  // 파라미터 타입에 따른 렌더링
  const renderParamInput = () => {
    const key = currentParam.key;
    const disabled = isTraining || currentParam.disabled;

    // COCO classes는 특별한 에디터 사용
    if (key === 'coco_classes') {
      return (
        <COCOClassesEditor
          value={getCurrentValue()}
          onChange={(value) => handleParamChange(key, value, currentParam)}
        />
      );
    }

    // boolean 타입
    if (currentParam.type === 'boolean') {
      return (
        <select
          value={getCurrentValue() ? 'true' : 'false'}
          onChange={(e) => handleParamChange(key, e.target.value === 'true', currentParam)}
          disabled={disabled}
          className={styles.select}
        >
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      );
    }

    // select 타입
    if (currentParam.options) {
      return (
        <select
          value={getCurrentValue()}
          onChange={(e) => handleParamChange(key, e.target.value, currentParam)}
          disabled={disabled}
          className={styles.select}
        >
          {currentParam.options.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      );
    }

    // slider 타입
    if (currentParam.type === 'slider') {
      return (
        <div className={styles.sliderContainer}>
          <Slider
            value={getCurrentValue()}
            onChange={(_, newValue) => handleParamChange(key, newValue, currentParam)}
            min={currentParam.min}
            max={currentParam.max}
            step={getSliderStep(currentParam)}
            disabled={disabled}
            valueLabelDisplay="auto"
            className={styles.slider}
          />
          <span className={styles.sliderValue}>{getCurrentValue()}</span>
        </div>
      );
    }

    // 기본 input 타입
    return (
      <input
        type={currentParam.type === 'float' ? 'number' : 'text'}
        value={getCurrentValue()}
        onChange={(e) => handleParamChange(key, e.target.value, currentParam)}
        disabled={disabled}
        className={styles.input}
        step={currentParam.type === 'float' ? '0.1' : undefined}
        min={currentParam.min}
        max={currentParam.max}
      />
    );
  };

  return (
    <div className={styles.paramEditor}>
      <div className={styles.paramHeader}>
        <label className={styles.paramLabel}>
          {currentParam.label}
          {currentParam.required && <span className={styles.required}>*</span>}
          {currentParam.description && (
            <HelpCircle size={16} className={styles.helpIcon} title={currentParam.description} />
          )}
        </label>
        {currentParam.unit && <span className={styles.unit}>({currentParam.unit})</span>}
      </div>
      
      {renderParamInput()}
      
      {paramErrors[currentParam.key] && (
        <div className={styles.errorMessage}>
          {paramErrors[currentParam.key]}
        </div>
      )}
      
      {currentParam.description && (
        <div className={styles.description}>
          {currentParam.description}
        </div>
      )}
    </div>
  );
};

export default ParameterEditor; 