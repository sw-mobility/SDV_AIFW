import React, { useState } from 'react';
import LabelingParameterSelector from './LabelingParameterSelector.jsx';
import LabelingParameterEditor from './LabelingParameterEditor.jsx';
import { LABELING_PARAM_GROUPS } from '../../../domain/labeling/labelingParameters.js';
import styles from './LabelingParameterSection.module.css';

/**
 * Labeling 파라미터 설정 섹션
 * Validation 페이지의 ValidationParameterSection과 정확히 동일한 구조
 * 좌측: 파라미터 그룹 선택기
 * 우측: 선택된 파라미터 편집기
 */
const LabelingParameterSection = ({
  labelingParams,
  onParamChange,
  onReset,
  selectedParamKeys,
  setSelectedParamKeys,
  disabled = false
}) => {
  const [openParamGroup, setOpenParamGroup] = useState(null);

  // 파라미터 키 토글
  const handleToggleParamKey = (paramKey) => {
    setSelectedParamKeys(prev => {
      if (prev.includes(paramKey)) {
        return prev.filter(key => key !== paramKey);
      } else {
        return [...prev, paramKey];
      }
    });
  };

  // 파라미터 키 제거
  const handleRemoveParamKey = (paramKey) => {
    setSelectedParamKeys(prev => prev.filter(key => key !== paramKey));
  };

  // 그룹 토글
  const handleToggleGroup = (groupIndex) => {
    setOpenParamGroup(openParamGroup === groupIndex ? null : groupIndex);
  };

  // 선택된 파라미터들 렌더링
  const renderParameterEditors = () => {
    if (selectedParamKeys.length === 0) {
      return (
        <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
          <span className={styles.emptyMessage}>왼쪽에서 파라미터를 선택하세요.</span>
        </div>
      );
    }

    return selectedParamKeys.map((key) => {
      // 파라미터 정의 찾기
      let foundParam = null;
      for (const group of LABELING_PARAM_GROUPS) {
        for (const param of group.params) {
          if (param.key === key) {
            foundParam = param;
            break;
          }
        }
        if (foundParam) break;
      }
      
      if (!foundParam) return null;
      
      return (
        <LabelingParameterEditor
          key={key}
          currentParam={foundParam}
          labelingParams={labelingParams}
          onParamChange={onParamChange}
          disabled={disabled}
        />
      );
    });
  };

  return (
    <div className={styles.paramSectionWrap}>
      {/* Left: Parameter Selector */}
      <div className={styles.paramSummaryBox}>
        <LabelingParameterSelector
          paramGroups={LABELING_PARAM_GROUPS}
          selectedParamKeys={selectedParamKeys}
          openParamGroup={openParamGroup}
          onToggleParamKey={handleToggleParamKey}
          onRemoveParamKey={handleRemoveParamKey}
          onToggleGroup={handleToggleGroup}
          onReset={onReset}
          disabled={disabled}
        />
      </div>
      
      {/* Center: Parameter Input Fields */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
    </div>
  );
};

export default LabelingParameterSection;
