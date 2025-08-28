import React from 'react';
import OptimizationParameterSelector from './OptimizationParameterSelector.jsx';
import OptimizationParameterEditor from './OptimizationParameterEditor.jsx';
import { getOptimizationParameterGroups } from '../../../domain/optimization/optimizationParameters.js';
import styles from './OptimizationParameterSection.module.css';

/**
 * Optimization 파라미터 설정 섹션
 * Validation 페이지와 동일한 구조
 */
const OptimizationParameterSection = ({
  optimizationType,
  optimizationParams,
  onParamChange,
  isRunning = false
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
        <div className={styles.paramCardEmpty}>
          <div className={styles.emptyMessage}>
            Select parameters from the left panel to configure optimization settings.
          </div>
        </div>
      );
    }

    return selectedParamKeys.map((paramKey) => {
      let foundParam = null;
      
      // 모든 그룹에서 파라미터 찾기
      const paramGroups = getOptimizationParameterGroups(optimizationType);
      for (const group of paramGroups) {
        for (const param of group.params) {
          if (param.key === paramKey) {
            foundParam = param;
            break;
          }
        }
        if (foundParam) break;
      }
      
      if (!foundParam) return null;

      return (
        <OptimizationParameterEditor
          key={paramKey}
          currentParam={foundParam}
          optimizationParams={optimizationParams}
          onParamChange={onParamChange}
          isRunning={isRunning}
        />
      );
    });
  };

  // 최적화 타입이 변경되면 선택된 파라미터 초기화
  React.useEffect(() => {
    setSelectedParamKeys([]);
    setOpenParamGroup(null);
  }, [optimizationType]);

  const paramGroups = getOptimizationParameterGroups(optimizationType);

  return (
    <div className={styles.paramSectionWrap}>
      {/* Left: Parameter Selector */}
      <div className={styles.paramSummaryBox}>
        <OptimizationParameterSelector
          paramGroups={paramGroups}
          selectedParamKeys={selectedParamKeys}
          openParamGroup={openParamGroup}
          onToggleParamKey={handleToggleParamKey}
          onRemoveParamKey={handleRemoveParamKey}
          onToggleGroup={handleToggleGroup}
          disabled={isRunning}
          optimizationType={optimizationType}
        />
      </div>
      
      {/* Right: Parameter Input Fields */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
    </div>
  );
};

export default OptimizationParameterSection;
