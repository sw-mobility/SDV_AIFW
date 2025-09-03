import React from 'react';
import ValidationParameterSelector from './ValidationParameterSelector.jsx';
import ValidationParameterEditor from './ValidationParameterEditor.jsx';
import { VALIDATION_PARAM_GROUPS } from '../../../domain/validation/validationParameters.js';
import styles from './ValidationParameterSection.module.css';

/**
 * Validation 파라미터 설정 섹션
 * Training 페이지의 ParameterSection과 정확히 동일한 구조
 * 좌측: 파라미터 그룹 선택기
 * 우측: 선택된 파라미터 편집기
 */
const ValidationParameterSection = ({
  validationParams,
  onParamChange,
  onReset,
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

  // TID 파라미터 (항상 표시)
  const tidParam = {
    key: 'tid',
    label: 'Training ID',
    type: 'tid', // ValidationParameterEditor에서 특별 처리
    required: true,
    desc: 'Training ID (e.g., T0001) for the model to validate',
    placeholder: 'T0001'
  };

  // 선택된 파라미터들 렌더링
  const renderParameterEditors = () => {
    const editors = [];
    
    // TID 파라미터는 항상 첫 번째로 표시
    editors.push(
      <ValidationParameterEditor
        key="tid"
        currentParam={tidParam}
        validationParams={validationParams}
        onParamChange={onParamChange}
        disabled={disabled}
      />
    );
    
    // 선택된 파라미터들 추가
    selectedParamKeys.forEach((key) => {
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
      
      if (!foundParam) return;
      
      editors.push(
        <ValidationParameterEditor
          key={key}
          currentParam={foundParam}
          validationParams={validationParams}
          onParamChange={onParamChange}
          disabled={disabled}
        />
      );
    });
    
    if (editors.length === 1) {
      // TID만 있는 경우 안내 메시지 추가
      editors.push(
        <div key="empty-message" className={styles.paramCard + ' ' + styles.paramCardEmpty}>
          <span className={styles.emptyMessage}>왼쪽에서 추가 파라미터를 선택하세요.</span>
        </div>
      );
    }
    
    return editors;
  };

  return (
    <div className={styles.paramSectionWrap}>
      {/* Left: Parameter Selector */}
      <div className={styles.paramSummaryBox}>
        <ValidationParameterSelector
          paramGroups={VALIDATION_PARAM_GROUPS}
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

export default ValidationParameterSection;
