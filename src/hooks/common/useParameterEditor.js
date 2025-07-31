import { useCallback } from 'react';
import { normalizeParamValue } from '../../domain/training/parameterGroups.js';

export const useParameterEditor = (parameters, onChange) => {
  // 파라미터 값 변경 처리
  const handleParamChange = useCallback((key, value, param) => {
    const newValue = normalizeParamValue(value, param);
    onChange(key, newValue, param);
  }, [onChange]);

  // 슬라이더용 step 값 계산
  const getSliderStep = useCallback((param) => {
    if (param.step) return param.step;
    
    // step이 정의되지 않은 경우 범위에 따라 적절한 값 설정
    const range = param.max - param.min;
    if (range <= 1) return 0.01;
    if (range <= 10) return 0.1;
    if (range <= 100) return 1;
    if (range <= 1000) return 10;
    return Math.ceil(range / 100);
  }, []);

  // 파라미터 값 가져오기 (기본값 포함)
  const getParamValue = useCallback((param, currentParams) => {
    return currentParams[param.key] ?? param.default;
  }, []);

  // 파라미터가 유효한지 확인
  const isParamValid = useCallback((param) => {
    return param && param.key && param.label;
  }, []);

  return {
    handleParamChange,
    getSliderStep,
    getParamValue,
    isParamValid
  };
}; 