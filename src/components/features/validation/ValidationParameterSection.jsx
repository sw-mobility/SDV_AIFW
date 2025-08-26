import React from 'react';
import ValidationParameterSelector from './ValidationParameterSelector.jsx';
import ValidationParameterEditor from './ValidationParameterEditor.jsx';
import { VALIDATION_PARAM_GROUPS } from '../../../domain/validation/validationParameters.js';
import styles from './ValidationParameterSection.module.css';

/**
 * Validation 파라미터 설정 섹션
 * Training 페이지의 ParameterSection 스타일을 참고하여 제작
 * 좌측: 파라미터 그룹 선택기
 * 우측: 선택된 파라미터 편집기
 */
const ValidationParameterSection = ({
  validationParams,
  onParamChange,
  disabled = false
}) => {
  const [selectedParamKeys, setSelectedParamKeys] = React.useState([]);
  const [openParamGroup, setOpenParamGroup] = React.useState(null);

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
          <span className={styles.emptyMessage}>
            왼쪽에서 파라미터를 선택하세요.
          </span>
        </div>
      );
    }

    return selectedParamKeys.map((key) => {
      // 파라미터 정의 찾기
      let foundParam = null;
      for (const group of VALIDATION_PARAM_GROUPS) {
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
        <ValidationParameterEditor
          key={key}
          currentParam={foundParam}
          validationParams={validationParams}
          onParamChange={onParamChange}
          disabled={disabled}
        />
      );
    });
  };

  return (
    <div className={styles.paramSectionWrap}>
      {/* 좌측: 파라미터 선택기 */}
      <div className={styles.paramSummaryBox}>
        <ValidationParameterSelector
          paramGroups={VALIDATION_PARAM_GROUPS}
          selectedParamKeys={selectedParamKeys}
          openParamGroup={openParamGroup}
          onToggleParamKey={handleToggleParamKey}
          onRemoveParamKey={handleRemoveParamKey}
          onToggleGroup={handleToggleGroup}
          disabled={disabled}
        />
      </div>
      
      {/* 우측: 파라미터 편집기 */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
    </div>
  );
};

export default ValidationParameterSection;
