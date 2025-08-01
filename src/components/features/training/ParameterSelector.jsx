import React from 'react';
import { ChevronDown, ChevronUp, Info, X } from 'lucide-react';
import { PARAMETER_GROUPS, PROJECT_INFO_PARAMS } from '../../../domain/training/parameterGroups.js';
import styles from './ParameterSelector.module.css';

const ParameterSelector = ({
  paramGroups,
  selectedParamKeys,
  openParamGroup,
  onToggleParamKey,
  onRemoveParamKey,
  onToggleGroup
}) => {
  const renderParameterChips = () => {
    if (selectedParamKeys.length === 0) return null;

    return (
      <div className={styles.selectedParamChips}>
        {selectedParamKeys.map((key) => {
          let foundParam = null;
          let isProjectInfo = false;
          
          for (const group of paramGroups) {
            for (const param of group.params) {
              if (param.key === key) {
                foundParam = param;
                isProjectInfo = group.group === PARAMETER_GROUPS.PROJECT_INFO;
                break;
              }
            }
            if (foundParam) break;
          }
          
          if (!foundParam) return null;
          
          return (
            <span
              key={key}
              className={`${styles.paramChip} ${isProjectInfo ? styles.projectInfo : ''}`}
            >
              {foundParam.label}
              <X 
                size={16} 
                className={styles.removeIcon} 
                onClick={() => onRemoveParamKey(key)} 
              />
            </span>
          );
        })}
      </div>
    );
  };

  const renderParameterGroup = (group, groupIndex) => {
    // Filter parameters for Project Information group
    let paramsToShow = group.params;
    if (group.group === PARAMETER_GROUPS.PROJECT_INFO) {
      paramsToShow = group.params.filter(p => PROJECT_INFO_PARAMS.includes(p.key));
    }
    
    if (paramsToShow.length === 0) return null;

    const isOpen = openParamGroup === groupIndex;
    const isProjectInfo = group.group === PARAMETER_GROUPS.PROJECT_INFO;

    return (
      <div key={group.group} className={styles.accordionCard}>
        <div
          className={`${styles.accordionHeader} ${isOpen ? styles.accordionOpen : ''}`}
          onClick={() => onToggleGroup(isOpen ? -1 : groupIndex)}
          tabIndex={0}
          role="button"
          aria-expanded={isOpen}
        >
          <span>{group.group}</span>
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
                    onClick={() => onToggleParamKey(param.key)}
                    className={`${styles.paramButton} ${isSelected ? styles.selected : ''} ${isProjectInfo ? styles.projectInfo : ''}`}
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
        )}
      </div>
    );
  };

  return (
    <div className={styles.parameterSelector}>
      <div className={styles.paramGroupTitle}>Parameters</div>
      
      {renderParameterChips()}
      
      <div className={styles.accordionGroups}>
        {paramGroups.map((group, index) => renderParameterGroup(group, index))}
      </div>
    </div>
  );
};

export default ParameterSelector; 