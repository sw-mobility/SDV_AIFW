import React from 'react';
import { ChevronDown, ChevronUp, Info, X, RotateCcw } from 'lucide-react';
import styles from './LabelingParameterSelector.module.css';

/**
 * Labeling 파라미터 그룹 선택 컴포넌트
 * Validation 페이지의 ValidationParameterSelector와 정확히 동일한 구조
 */
const LabelingParameterSelector = ({
  paramGroups,
  selectedParamKeys,
  openParamGroup,
  onToggleParamKey,
  onRemoveParamKey,
  onToggleGroup,
  onReset,
  disabled = false
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

    const isOpen = openParamGroup === groupIndex;

    return (
      <div key={groupIndex} className={styles.accordionCard}>
        <div
          className={`${styles.accordionHeader} ${isOpen ? styles.accordionOpen : ''}`}
          onClick={() => !disabled && onToggleGroup(isOpen ? -1 : groupIndex)}
          tabIndex={0}
          role="button"
          aria-expanded={isOpen}
        >
          <span>{group.title}</span>
          <span className={styles.accordionArrow}>
            {isOpen ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}
          </span>
        </div>
        
        {isOpen && (
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
                    {param.description && (
                      <Info 
                        size={15} 
                        color={isSelected ? '#fff' : '#888'} 
                        className={styles.infoIcon} 
                        title={param.description} 
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={styles.parameterSelector}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h3 className={styles.title}>Parameters</h3>
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
      
      <div className={styles.accordionContainer}>
        {paramGroups.map((group, index) => renderParameterGroup(group, index))}
      </div>
    </div>
  );
};

export default LabelingParameterSelector;
