import React from 'react';
import CodeEditor from '../../ui/editor/CodeEditor.jsx';
import ParameterEditor from './ParameterEditor.jsx';
import ExpertModeToggle from './ExpertModeToggle.jsx';
import ParameterSelector from './ParameterSelector.jsx';
import { TRAINING_TYPES } from '../../../domain/training/trainingTypes.js';
import styles from './ParameterSection.module.css';

const ParameterSection = ({
  // Expert mode state
  showCodeEditor,
  setShowCodeEditor,
  
  // Parameter state
  paramGroups,
  selectedParamKeys,
  openParamGroup,
  onToggleParamKey,
  onRemoveParamKey,
  onToggleGroup,
  
  // Snapshot state
  snapshots,
  selectedSnapshot,
  setSelectedSnapshot,
  editorFileStructure,
  editorFiles,
  
  // Parameter values
  algoParams,
  onParamChange,
  paramErrors,
  isTraining,
  
  // Training type
  trainingType
}) => {
  const renderSnapshotSelector = () => {
    if (!showCodeEditor) return null;

    const label = trainingType === TRAINING_TYPES.CONTINUAL 
      ? 'Select Base Snapshot (Required)' 
      : 'Select Snapshot';

    const placeholder = trainingType === TRAINING_TYPES.CONTINUAL
      ? 'Choose a base snapshot...'
      : 'Choose a snapshot...';

    return (
      <div className={styles.expertModeSection}>
        <label className={styles.paramLabel}>{label}</label>
        <select
          className={styles.select}
          value={selectedSnapshot?.id || ''}
          onChange={(e) => {
            const snapshot = snapshots.find(s => s.id === e.target.value);
            setSelectedSnapshot(snapshot);
          }}
        >
          <option value="">{placeholder}</option>
          {snapshots.map(snapshot => (
            <option key={snapshot.id} value={snapshot.id}>
              {snapshot.name}
            </option>
          ))}
        </select>
      </div>
    );
  };

  const renderCodeEditor = () => {
    if (!showCodeEditor) return null;

    return (
      <div className={styles.rightSection}>
        <div className={styles.codeEditorCard}>
          {selectedSnapshot ? (
            <CodeEditor
              snapshotName={selectedSnapshot.name}
              fileStructure={editorFileStructure}
              files={editorFiles}
              onSaveSnapshot={name => {
                alert(`Saved as snapshot: ${name}`);
              }}
              onCloseDrawer={() => setShowCodeEditor(false)}
            />
          ) : (
            <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
              <span className={styles.emptyMessage}>
                {trainingType === TRAINING_TYPES.CONTINUAL 
                  ? '왼쪽에서 기본 스냅샷을 선택하세요.'
                  : '왼쪽에서 스냅샷을 선택하세요.'
                }
              </span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderParameterEditors = () => {
    if (selectedParamKeys.length === 0) {
      return (
        <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
          <span className={styles.emptyMessage}>왼쪽에서 파라미터를 선택하세요.</span>
        </div>
      );
    }

    return selectedParamKeys.map((key) => {
      // Find parameter definition
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
        <ParameterEditor
          key={key}
          currentParam={foundParam}
          algoParams={algoParams}
          onParamChange={onParamChange}
          paramErrors={paramErrors}
          isTraining={isTraining}
        />
      );
    });
  };

  return (
    <div className={`${styles.paramSectionWrap} ${showCodeEditor ? styles.expertMode : ''}`}>
      {/* Left: Parameter Selector + Expert Mode */}
      <div className={styles.paramSummaryBox}>
        <ParameterSelector
          paramGroups={paramGroups}
          selectedParamKeys={selectedParamKeys}
          openParamGroup={openParamGroup}
          onToggleParamKey={onToggleParamKey}
          onRemoveParamKey={onRemoveParamKey}
          onToggleGroup={onToggleGroup}
          trainingType={trainingType}
        />
        
        <ExpertModeToggle
          isActive={showCodeEditor}
          onToggle={() => setShowCodeEditor(!showCodeEditor)}
        />
        
        {renderSnapshotSelector()}
      </div>
      
      {/* Center: Parameter Input Fields */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
      
      {/* Right: Code Editor (Expert Mode only) */}
      {renderCodeEditor()}
    </div>
  );
};

export default ParameterSection; 