import React from 'react';
import CodeEditor from '../ui/editor/CodeEditor.jsx';
import CodeEditorSkeleton from '../ui/editor/CodeEditorSkeleton.jsx';
import { useCodeEditor } from '../../hooks/editor/useCodeEditor.js';
import styles from './ParameterSection.module.css';

// Type별 컴포넌트 import
import TrainingParameterSelector from '../features/training/ParameterSelector.jsx';
import TrainingParameterEditor from '../features/training/ParameterEditor.jsx';
import ValidationParameterSelector from '../features/validation/ValidationParameterSelector.jsx';
import ValidationParameterEditor from '../features/validation/ValidationParameterEditor.jsx';
import OptimizationParameterSelector from '../features/optimization/OptimizationParameterSelector.jsx';
import OptimizationParameterEditor from '../features/optimization/OptimizationParameterEditor.jsx';

// Domain imports
import { VALIDATION_PARAM_GROUPS } from '../../domain/validation/validationParameters.js';
import { getOptimizationParameterGroups } from '../../domain/optimization/optimizationParameters.js';

/**
 * 통합된 ParameterSection 컴포넌트
 * Training, Validation, Optimization의 공통 파라미터 설정 섹션
 * 
 * @param {string} type - 'training' | 'validation' | 'optimization'
 * @param {Object} props - 타입별 props
 */
const ParameterSection = ({ type, ...props }) => {
  // 공통 상태 관리
  const [selectedParamKeys, setSelectedParamKeys] = React.useState([]);
  const [openParamGroup, setOpenParamGroup] = React.useState(null);

  // 공통 핸들러들
  const handleToggleParamKey = (paramKey) => {
    setSelectedParamKeys(prev => {
      if (prev.includes(paramKey)) {
        return prev.filter(key => key !== paramKey);
      } else {
        return [...prev, paramKey];
      }
    });
  };

  const handleRemoveParamKey = (paramKey) => {
    setSelectedParamKeys(prev => prev.filter(key => key !== paramKey));
  };

  const handleToggleGroup = (groupIndex) => {
    setOpenParamGroup(openParamGroup === groupIndex ? null : groupIndex);
  };

  // 타입별 렌더링 함수들
  const renderParameterSelector = () => {
    switch (type) {
      case 'training':
        return (
          <TrainingParameterSelector
            paramGroups={props.paramGroups}
            selectedParamKeys={selectedParamKeys}
            openParamGroup={openParamGroup}
            onToggleParamKey={handleToggleParamKey}
            onRemoveParamKey={handleRemoveParamKey}
            onToggleGroup={handleToggleGroup}
            onReset={props.onReset}
            trainingType={props.trainingType}
          />
        );
      case 'validation':
        return (
          <ValidationParameterSelector
            paramGroups={VALIDATION_PARAM_GROUPS}
            selectedParamKeys={selectedParamKeys}
            openParamGroup={openParamGroup}
            onToggleParamKey={handleToggleParamKey}
            onRemoveParamKey={handleRemoveParamKey}
            onToggleGroup={handleToggleGroup}
            onReset={props.onReset}
            disabled={props.disabled}
          />
        );
      case 'optimization':
        const paramGroups = getOptimizationParameterGroups(props.optimizationType);
        return (
          <OptimizationParameterSelector
            paramGroups={paramGroups}
            selectedParamKeys={selectedParamKeys}
            openParamGroup={openParamGroup}
            onToggleParamKey={handleToggleParamKey}
            onRemoveParamKey={handleRemoveParamKey}
            onToggleGroup={handleToggleGroup}
            onReset={props.onReset}
            disabled={props.isRunning}
            optimizationType={props.optimizationType}
          />
        );
      default:
        return null;
    }
  };

  const renderParameterEditors = () => {
    const editors = [];
    
    // Validation의 경우 TID 파라미터를 항상 첫 번째로 표시
    if (type === 'validation') {
      const tidParam = {
        key: 'tid',
        label: 'Training ID',
        type: 'tid',
        required: true,
        desc: 'Training ID (e.g., T0001) for the model to validate',
        placeholder: 'T0001'
      };
      
      editors.push(
        <ValidationParameterEditor
          key="tid"
          currentParam={tidParam}
          validationParams={props.validationParams}
          onParamChange={props.onParamChange}
          disabled={props.disabled}
          projectId={props.projectId}
        />
      );
    }
    
    // 선택된 파라미터들 추가
    selectedParamKeys.forEach((key) => {
      let foundParam = null;
      let paramGroups = [];
      
      // 타입별 파라미터 그룹 설정
      switch (type) {
        case 'training':
          paramGroups = props.paramGroups || [];
          break;
        case 'validation':
          paramGroups = VALIDATION_PARAM_GROUPS;
          break;
        case 'optimization':
          paramGroups = getOptimizationParameterGroups(props.optimizationType);
          break;
      }
      
      // 파라미터 정의 찾기
      for (const group of paramGroups) {
        for (const param of group.params) {
          if (param.key === key) {
            foundParam = param;
            break;
          }
        }
        if (foundParam) break;
      }
      
      if (!foundParam) return;
      
      // 타입별 파라미터 에디터 렌더링
      switch (type) {
        case 'training':
          editors.push(
            <TrainingParameterEditor
              key={key}
              currentParam={foundParam}
              algoParams={props.algoParams}
              onParamChange={props.onParamChange}
              paramErrors={props.paramErrors}
              disabled={false}
              selectedDataset={props.selectedDataset}
            />
          );
          break;
        case 'validation':
          editors.push(
            <ValidationParameterEditor
              key={key}
              currentParam={foundParam}
              validationParams={props.validationParams}
              onParamChange={props.onParamChange}
              disabled={props.disabled}
              projectId={props.projectId}
            />
          );
          break;
        case 'optimization':
          editors.push(
            <OptimizationParameterEditor
              key={key}
              currentParam={foundParam}
              optimizationParams={props.optimizationParams}
              onParamChange={props.onParamChange}
              isRunning={props.isRunning}
              projectId={props.projectId}
            />
          );
          break;
      }
    });
    
    // 빈 상태 메시지
    if (editors.length === 0 || (type === 'validation' && editors.length === 1)) {
      const message = type === 'validation' && editors.length === 1 
        ? '왼쪽에서 추가 파라미터를 선택하세요.'
        : '왼쪽에서 파라미터를 선택하세요.';
        
      editors.push(
        <div key="empty-message" className={styles.paramCard + ' ' + styles.paramCardEmpty}>
          <span className={styles.emptyMessage}>{message}</span>
        </div>
      );
    }
    
    return editors;
  };

  // Codebase selector 렌더링 (Training과 Validation만)
  const renderCodebaseSelector = () => {
    if (type === 'optimization') return null;
    
    return (
      <div className={styles.expertModeSection}>
        <label className={styles.paramLabel}>Select Codebase (Optional)</label>
        <select
          className={styles.select}
          value={props.selectedCodebase?.cid || ''}
          onChange={(e) => {
            const codebase = props.codebases.find(c => c.cid === e.target.value);
            props.setSelectedCodebase(codebase);
            if (codebase) {
              props.setShowCodeEditor(true);
            } else {
              props.setShowCodeEditor(false);
            }
          }}
          disabled={(type === 'training' ? props.isTraining : props.isValidating) || props.codebaseLoading}
        >
          <option value="">코드베이스 사용 안함</option>
          {props.codebases && Array.isArray(props.codebases) && props.codebases.map(codebase => (
            <option key={codebase.cid} value={codebase.cid}>
              {codebase.name || codebase.cid}
            </option>
          ))}
        </select>
        {props.codebaseError && (
          <div className={styles.errorMessage}>
            {props.codebaseError}
          </div>
        )}
      </div>
    );
  };

  // Code editor 렌더링 (Training과 Validation만)
  const renderCodeEditor = () => {
    if (type === 'optimization' || !props.showCodeEditor) return null;
    
    // useCodeEditor hook 사용
    const {
      fileStructure,
      files,
      activeFile,
      loading: editorLoading,
      error: editorError,
      changeActiveFile,
      currentFile
    } = useCodeEditor(props.selectedCodebase);

    return (
      <div className={styles.rightSection}>
        <div className={styles.codeEditorCard}>
          {props.selectedCodebase ? (
            (editorLoading || props.codebaseFilesLoading) ? (
              <CodeEditorSkeleton compact={false} />
            ) : (
              <CodeEditor
                key={props.selectedCodebase?.cid}
                snapshotName={`${props.selectedCodebase.name || props.selectedCodebase.cid} (Preview)`}
                fileStructure={fileStructure}
                files={files}
                activeFile={activeFile}
                onFileChange={changeActiveFile}
                onFilesChange={() => {}}
                onSaveSnapshot={() => {}}
                onCloseDrawer={() => props.setShowCodeEditor(false)}
                compact={false}
                hideSaveButtons={true}
                currentFile={currentFile}
                readOnly={true}
                onEditorChange={() => {}}
                showPreviewMode={true}
              />
            )
          ) : (
            <div className={styles.paramCard + ' ' + styles.paramCardEmpty}>
              <span className={styles.emptyMessage}>
                왼쪽에서 코드베이스를 선택하세요.
              </span>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Optimization 타입 변경 시 파라미터 초기화
  React.useEffect(() => {
    if (type === 'optimization') {
      setSelectedParamKeys([]);
      const paramGroups = getOptimizationParameterGroups(props.optimizationType);
      if (paramGroups.length > 0) {
        setOpenParamGroup(0);
      }
    }
  }, [type, props.optimizationType]);

  return (
    <div className={props.showCodeEditor ? styles.paramSectionWrapExtended : styles.paramSectionWrap}>
      {/* Left: Parameter Selector */}
      <div className={styles.paramSummaryBox}>
        {renderParameterSelector()}
        {renderCodebaseSelector()}
      </div>
      
      {/* Center: Parameter Input Fields */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
      
      {/* Right: Code Editor (when applicable) */}
      {renderCodeEditor()}
    </div>
  );
};

export default ParameterSection;
