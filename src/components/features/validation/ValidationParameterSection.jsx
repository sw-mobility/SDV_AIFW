import React from 'react';
import ValidationParameterSelector from './ValidationParameterSelector.jsx';
import ValidationParameterEditor from './ValidationParameterEditor.jsx';
import { VALIDATION_PARAM_GROUPS } from '../../../domain/validation/validationParameters.js';
import CodeEditor from '../../ui/editor/CodeEditor.jsx';
import CodeEditorSkeleton from '../../ui/editor/CodeEditorSkeleton.jsx';
import { useCodeEditor } from '../../../hooks/editor/useCodeEditor.js';
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
  disabled = false,
  projectId = 'P0001',
  // Codebase 관련 props 추가 (training과 동일)
  codebases = [],
  selectedCodebase,
  setSelectedCodebase,
  codebaseLoading = false,
  codebaseError = null,
  codebaseFileStructure = {},
  codebaseFiles = {},
  codebaseFilesLoading = false,
  showCodeEditor = false,
  setShowCodeEditor,
  isValidating = false
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

  // Codebase selector 렌더링 (training과 동일)
  const renderCodebaseSelector = () => {
    return (
      <div className={styles.expertModeSection}>
        <label className={styles.paramLabel}>Select Codebase (Optional)</label>
        <select
          className={styles.select}
          value={selectedCodebase?.cid || ''}
          onChange={(e) => {
            const codebase = codebases.find(c => c.cid === e.target.value);
            setSelectedCodebase(codebase);
            if (codebase) { // If a codebase is selected (not "코드베이스 사용 안함")
              setShowCodeEditor(true);
            } else { // If "코드베이스 사용 안함" is selected
              setShowCodeEditor(false);
            }
          }}
          disabled={isValidating || codebaseLoading}
        >
          <option value="">코드베이스 사용 안함</option>
          {codebases && Array.isArray(codebases) && codebases.map(codebase => (
            <option key={codebase.cid} value={codebase.cid}>
              {codebase.name || codebase.cid}
            </option>
          ))}
        </select>
        {codebaseError && (
          <div className={styles.errorMessage}>
            {codebaseError}
          </div>
        )}
      </div>
    );
  };

  // useCodeEditor hook 사용 (Code Editor 페이지와 동일)
  const {
    fileStructure,
    files,
    activeFile,
    loading: editorLoading,
    error: editorError,
    changeActiveFile,
    currentFile
  } = useCodeEditor(selectedCodebase);

  // Code editor 렌더링 (training과 동일)
  const renderCodeEditor = () => {
    if (!showCodeEditor) return null;

    return (
      <div className={styles.rightSection}>
        <div className={styles.codeEditorCard}>
          {selectedCodebase ? (
            (editorLoading || codebaseFilesLoading) ? (
              <CodeEditorSkeleton compact={false} />
            ) : (
              <CodeEditor
                key={selectedCodebase?.cid} // 코드베이스가 변경될 때만 리렌더링
                snapshotName={`${selectedCodebase.name || selectedCodebase.cid} (Preview)`}
                fileStructure={fileStructure}
                files={files}
                activeFile={activeFile}
                onFileChange={changeActiveFile}
                onFilesChange={() => {}} // Read-only
                onSaveSnapshot={() => {}} // 저장 기능 완전 비활성화
                onCloseDrawer={() => setShowCodeEditor(false)}
                compact={false} // Display file tree
                hideSaveButtons={true}
                currentFile={currentFile} // 최신 currentFile 전달
                readOnly={true} // 수정 불가
                onEditorChange={() => {}} // 수정 이벤트 무시
                showPreviewMode={true} // Preview 모드 표시
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
        projectId={projectId}
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
          projectId={projectId}
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
    <div className={showCodeEditor ? styles.paramSectionWrapExtended : styles.paramSectionWrap}>
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
        
        {/* Codebase Selector 추가 (training과 동일한 위치) */}
        {renderCodebaseSelector()}
      </div>
      
      {/* Center: Parameter Input Fields */}
      <div className={styles.paramCardWrap}>
        {renderParameterEditors()}
      </div>
      
      {/* Right: Code Editor (when codebase is selected) */}
      {renderCodeEditor()}
    </div>
  );
};

export default ValidationParameterSection;
