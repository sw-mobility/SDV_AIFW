import React from 'react';
import CodeEditor from '../../ui/editor/CodeEditor.jsx';
import ParameterEditor from './ParameterEditor.jsx';
import ExpertModeToggle from './ExpertModeToggle.jsx';
import ParameterSelector from './ParameterSelector.jsx';
import { TRAINING_TYPES } from '../../../domain/training/trainingTypes.js';
import styles from './ParameterSection.module.css';

/**
 * training page 의 중앙 삼단구조 좌측필드
 * 알고리즘 파라미터 설정 섹션
 * 주요 기능:
 * 파라미터 그룹별 설정
 * 전문가 모드 토글
 * 파라미터 편집기 제공
 *
 * @param showCodeEditor
 * @param setShowCodeEditor
 * @param paramGroups
 * @param selectedParamKeys
 * @param openParamGroup
 * @param onToggleParamKey
 * @param onRemoveParamKey
 * @param onToggleGroup
 * @param snapshots
 * @param selectedSnapshot
 * @param setSelectedSnapshot
 * @param editorFileStructure
 * @param editorFiles
 * @param algoParams
 * @param onParamChange
 * @param paramErrors
 * @param isTraining
 * @param trainingType
 * @returns {Element}
 * @constructor
 */
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
  onReset,
  
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
  trainingType,
  selectedDataset
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
    const editors = [];
    
    // 선택된 파라미터들 추가
    selectedParamKeys.forEach((key) => {
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
      
      if (foundParam) {
        editors.push(
          <ParameterEditor
            key={key}
            currentParam={foundParam}
            algoParams={algoParams}
            onParamChange={onParamChange}
            paramErrors={paramErrors}
            disabled={false}
            selectedDataset={selectedDataset}
          />
        );
      }
    });
    
    if (editors.length === 0) {
      // 파라미터가 선택되지 않은 경우 안내 메시지 추가
      editors.push(
        <div key="empty-message" className={styles.paramCard + ' ' + styles.paramCardEmpty}>
          <span className={styles.emptyMessage}>왼쪽에서 파라미터를 선택하세요.</span>
        </div>
      );
    }
    
    return editors;
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
          onReset={onReset}
          trainingType={trainingType}
        />
        
        {/* Expert Mode Toggle - UI에서 숨김 */}
        {/* <ExpertModeToggle
          isActive={showCodeEditor}
          onToggle={() => setShowCodeEditor(!showCodeEditor)}
        /> */}
        
        {/* Snapshot Selector - UI에서 숨김 */}
        {/* {renderSnapshotSelector()} */}
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