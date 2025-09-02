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
  onReset,
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

  // Model ID 파라미터 (항상 표시)
  const modelIdParam = {
    key: 'model_id',
    label: 'Model ID',
    type: 'text',
    required: true,
    desc: 'Training ID (e.g., T0001) or Optimizing ID (e.g., O0001) for the model to optimize',
    placeholder: 'T0001 or O0001'
  };

  // 선택된 파라미터들 렌더링
  const renderParameterEditors = () => {
    const editors = [];
    
    // Model ID는 항상 첫 번째에 표시
    editors.push(
      <OptimizationParameterEditor
        key="model_id"
        currentParam={modelIdParam}
        optimizationParams={optimizationParams}
        onParamChange={onParamChange}
        isRunning={isRunning}
      />
    );
    
    // 선택된 다른 파라미터들 추가
    selectedParamKeys.forEach((paramKey) => {
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
      
      if (foundParam) {
        editors.push(
          <OptimizationParameterEditor
            key={paramKey}
            currentParam={foundParam}
            optimizationParams={optimizationParams}
            onParamChange={onParamChange}
            isRunning={isRunning}
          />
        );
      }
    });
    
    return editors;
  };

  // 최적화 타입이 변경되면 선택된 파라미터 초기화하고 모든 그룹을 펼침
  React.useEffect(() => {
    setSelectedParamKeys([]);
    // 모든 그룹을 펼친 상태로 설정
    const paramGroups = getOptimizationParameterGroups(optimizationType);
    if (paramGroups.length > 0) {
      setOpenParamGroup(0); // 첫 번째 그룹을 펼침
    }
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
          onReset={onReset}
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
