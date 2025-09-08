import React, { useState, useEffect, useCallback, useRef } from 'react';
import CodeEditor from '../../components/ui/editor/CodeEditor.jsx';
import CodeEditorSkeleton from '../../components/ui/editor/CodeEditorSkeleton.jsx';
import CodebaseManager from '../../components/features/editor/CodebaseManager.jsx';
import CodebaseInfo from '../../components/features/editor/CodebaseInfo.jsx';
import Toast from '../../components/ui/atoms/Toast.jsx';
import { useCodebaseManager } from '../../hooks/editor/useCodebaseManager.js';
import { useCodeEditor } from '../../hooks/editor/useCodeEditor.js';
import Button from '../../components/ui/atoms/Button.jsx';
import ErrorMessage from '../../components/ui/atoms/ErrorMessage.jsx';
import { Save, RefreshCw, AlertCircle } from 'lucide-react';
import styles from './EditorPage.module.css';

const EditorPage = () => {
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastType, setToastType] = useState('success');
  
  // Monaco Editor 인스턴스 참조
  const monacoEditorRef = useRef(null);

  // 코드베이스 관리
  const {
    codebases,
    selectedCodebase,
    loading: codebaseLoading,
    error: codebaseError,
    isCreating,
    handleCreateCodebase,
    handleUpdateCodebase,
    handleDeleteCodebase,
    handleSelectCodebase,
    loadCodebase
  } = useCodebaseManager();

  // 코드 에디터 (선택된 코드베이스 기반)
  const {
    fileStructure,
    files,
    activeFile,
    loading: editorLoading,
    error: editorError,
    hasUnsavedChanges,
    lastSavedAt,
    currentFile,
    isEmpty,
    updateFileContent,
    updateFileLanguage,
    changeActiveFile,
    saveChanges,
    discardChanges,
    createNewFile,
    loadCodeTemplate
  } = useCodeEditor(selectedCodebase);

  // Handle file content change in editor
  const handleEditorChange = useCallback((value) => {
    if (activeFile && value !== undefined) {
      updateFileContent(activeFile, value);
    }
  }, [activeFile, updateFileContent]);

  // Toast 메시지 표시 헬퍼
  const showToastMessage = useCallback((message, type = 'success') => {
    setToastMessage(message);
    setToastType(type);
    setShowToast(true);
  }, []);

  // Monaco Editor 마운트 핸들러
  const handleEditorMount = useCallback((editor) => {
    monacoEditorRef.current = editor;
  }, []);

  // Handle language change in editor
  const handleLanguageChange = (language) => {
    if (activeFile) {
      updateFileLanguage(activeFile, language);
    }
  };

  // Handle save action
  const handleSave = useCallback(async () => {
    if (!selectedCodebase) {
      showToastMessage('Please select a codebase to save.', 'error');
      return;
    }


    // Monaco Editor의 실제 값을 React 상태와 동기화
    if (monacoEditorRef.current && activeFile) {
      const editorValue = monacoEditorRef.current.getValue();
      const currentFileContent = files[activeFile]?.code;
      
      // 에디터 값과 React 상태가 다르면 동기화
      if (editorValue !== currentFileContent) {
        updateFileContent(activeFile, editorValue);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    try {
      const result = await saveChanges(selectedCodebase.cid);
      if (result.success) {
        showToastMessage('Changes saved successfully!', 'success');
      } else {
        showToastMessage(result.message || 'Failed to save changes.', 'error');
      }
    } catch (error) {
      console.error('Save error:', error);
      showToastMessage('An error occurred while saving.', 'error');
    }
  }, [selectedCodebase, files, saveChanges, showToastMessage, monacoEditorRef, activeFile, updateFileContent]);

  // Handle codebase create
  const handleCodebaseCreate = async (requestData) => {
    try {
      await handleCreateCodebase(requestData);
      showToastMessage('Codebase created and template loaded successfully!', 'success');
    } catch (error) {
      showToastMessage('Failed to create codebase.', 'error');
      throw error;
    }
  };

  // Handle codebase update
  const handleCodebaseUpdate = async (cid, requestData) => {
    try {
      await handleUpdateCodebase(cid, requestData);
      showToastMessage('Codebase updated successfully!', 'success');
    } catch (error) {
      showToastMessage('Failed to update codebase.', 'error');
      throw error;
    }
  };

  // Handle codebase delete
  const handleCodebaseDelete = async (cid) => {
    try {
      await handleDeleteCodebase(cid);
      showToastMessage('Codebase deleted successfully!', 'success');
    } catch (error) {
      showToastMessage('Failed to delete codebase.', 'error');
      throw error;
    }
  };



  return (
      <div className={styles.pageContainer}>
        <div className={styles.container}>
          <div className={styles.pageHeader}>
            <h1 className={styles.pageTitle}>Code Editor</h1>
            <p className={styles.pageDescription}>
              A comprehensive development environment for managing and editing codebases with full CRUD operations
            </p>
          </div>

          {/* Error Display */}
          {(codebaseError || editorError) && (
              <div className={styles.errorSection}>
                {codebaseError && <ErrorMessage message={`Codebase Error: ${codebaseError}`} />}
                {editorError && <ErrorMessage message={`Editor Error: ${editorError}`} />}
              </div>
          )}
        </div>

        {/* Main Editor Section */}
        <div className={styles.editorSection}>
          <div className={styles.editorLayout}>
            {/* Left: Codebase Management Panel */}
            <div className={styles.sidebar}>
              <CodebaseManager
                  codebases={codebases}
                  selectedCodebase={selectedCodebase}
                  onCodebaseSelect={handleSelectCodebase}
                  onCodebaseCreate={handleCodebaseCreate}
                  onCodebaseUpdate={handleCodebaseUpdate}
                  onCodebaseDelete={handleCodebaseDelete}
                  loading={codebaseLoading}
              />
            </div>

            {/* Right: Editor Area */}
            <div className={styles.editorMain}>
              {/* Codebase Info Header */}
              <div className={styles.editorHeader}>
                <CodebaseInfo
                    codebase={selectedCodebase}
                    loading={codebaseLoading}
                    files={files}
                    lastSavedAt={lastSavedAt}
                />

                {/* Toolbar */}
                <div className={styles.toolbar}>
                  <div className={styles.toolbarLeft}>
                    {selectedCodebase && (
                        <div className={styles.codebaseStatus}>
                      <span className={styles.statusText}>
                        {selectedCodebase.name || selectedCodebase.cid}
                      </span>
                          {hasUnsavedChanges && (
                              <span className={styles.unsavedIndicator}>
                          <AlertCircle size={14} />
                          Unsaved Changes
                        </span>
                          )}
                        </div>
                    )}
                  </div>

                  <div className={styles.toolbarRight}>
                    <Button
                        onClick={handleSave}
                        variant="primary"
                        size="medium"
                        disabled={!selectedCodebase || !hasUnsavedChanges}
                    >
                      Save Changes
                    </Button>
                  </div>
                </div>
              </div>

              {/* Code Editor */}
              <div className={styles.editorContainer}>
                {!selectedCodebase ? (
                    <div className={styles.emptyState}>
                      <h3>Select a Codebase</h3>
                      <p>Choose a codebase from the left panel or create a new one to get started.</p>
                    </div>
                ) : (editorLoading || codebaseLoading || isCreating) ? (
                    <CodeEditorSkeleton compact={false} />
                ) : isEmpty ? (
                    <div className={styles.emptyState}>
                      <h3>No Code Files</h3>
                      <p>The selected codebase contains no files.</p>
                    </div>
                ) : (
                    <CodeEditor
                        key={selectedCodebase?.cid} // 코드베이스가 변경될 때만 리렌더링
                        fileStructure={fileStructure}
                        files={files}
                        activeFile={activeFile}
                        onFileChange={changeActiveFile}
                        onFilesChange={() => {}} // 사용하지 않음
                        onSaveSnapshot={handleSave}
                        snapshotName={selectedCodebase.name || selectedCodebase.cid}
                        compact={false}
                        hideSaveButtons={true}
                        currentFile={currentFile} // 최신 currentFile 전달
                        onEditorChange={handleEditorChange}
                        onLanguageChange={handleLanguageChange}
                        onSnapshotSave={handleSave}
                        onEditorMount={handleEditorMount} // 에디터 인스턴스 전달
                    />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Toast Notification */}
        <Toast
            message={toastMessage}
            type={toastType}
            isVisible={showToast}
            onClose={() => setShowToast(false)}
            duration={4000}
        />
      </div>
  );
};

export default EditorPage;