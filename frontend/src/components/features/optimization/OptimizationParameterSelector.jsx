import React from 'react';
import { ChevronDown, ChevronUp, Info, X, RotateCcw } from 'lucide-react';
import styles from './OptimizationParameterSelector.module.css';

/**
 * Optimization 파라미터 그룹 선택 컴포넌트
 * Training/Validation 페이지의 ParameterSelector와 정확히 동일한 구조
 */
const OptimizationParameterSelector = ({
  paramGroups,
  selectedParamKeys,
  openParamGroup,
  onToggleParamKey,
  onRemoveParamKey,
  onToggleGroup,
  onReset,
  disabled = false,
  optimizationType
}) => {
  const renderParameterChips = () => {
    if (selectedParamKeys.length === 0) return null;

    return (
      <div className={styles.selectedParamChips}>
        {selectedParamKeys.map((key) => {
          let foundParam = null;
          
          for (const group of paramGroups) {
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
            <span
              key={key}
              className={styles.paramChip}
            >
              {foundParam.label}
              <X 
                size={16} 
                className={styles.removeIcon} 
                onClick={() => !disabled && onRemoveParamKey(key)} 
              />
            </span>
          );
        })}
      </div>
    );
  };

  const renderParameterGroup = (group, groupIndex) => {
    // Show all parameters in the group
    const paramsToShow = group.params;
    
    if (paramsToShow.length === 0) return null;

    // 항상 펼쳐진 상태로 설정
    const isOpen = true;

    return (
      <div key={group.group} className={styles.accordionCard}>
        <div
          className={`${styles.accordionHeader} ${styles.accordionOpen}`}
          tabIndex={0}
          role="button"
          aria-expanded={true}
        >
          <span>{group.group}</span>
          <span className={styles.accordionArrow}>
            <ChevronUp size={18} color="#4f8cff" />
          </span>
        </div>
        
        <div className={styles.accordionContent}>
          <div className={styles.paramButtons}>
            {paramsToShow.map((param) => {
              const isSelected = selectedParamKeys.includes(param.key);
              
              return (
                <button
                  key={param.key}
                  type="button"
                  onClick={() => !disabled && onToggleParamKey(param.key)}
                  className={`${styles.paramButton} ${isSelected ? styles.selected : ''}`}
                  disabled={disabled}
                >
                  {param.label}
                  {param.desc && (
                    <Info 
                      size={15} 
                      color={isSelected ? '#fff' : '#888'} 
                      className={styles.infoIcon} 
                      title={param.desc} 
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={styles.parameterSelector}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <div className={styles.paramGroupTitle}>Parameters</div>
        </div>
        <div className={styles.actions}>
          <button
            type="button"
            onClick={onReset}
            className={styles.resetButton}
            disabled={disabled}
            title="Reset to default"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>
      
      {renderParameterChips()}
      
      <div className={styles.accordionGroups}>
        {paramGroups.map((group, index) => renderParameterGroup(group, index))}
      </div>
    </div>
  );
};

export default OptimizationParameterSelector;

