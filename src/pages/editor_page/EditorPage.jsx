import React, { useState, useEffect } from 'react';
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
  
  // 코드베이스 관리
  const {
    codebases,
    selectedCodebase,
    loading: codebaseLoading,
    error: codebaseError,
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
  } = useCodeEditor(selectedCodebase?.algorithm);

  // 코드베이스 선택 시 해당 코드베이스 로드
  useEffect(() => {
    if (selectedCodebase) {
      loadCodebase(selectedCodebase.cid)
        .then((data) => {
          // 코드베이스 데이터를 에디터에 로드
          loadCodeTemplate(selectedCodebase.algorithm);
        })
        .catch((error) => {
          console.error('Failed to load codebase:', error);
          showToastMessage('Failed to load codebase.', 'error');
        });
    }
  }, [selectedCodebase, loadCodebase, loadCodeTemplate]);

  // Handle file content change in editor
  const handleEditorChange = (value) => {
    if (activeFile) {
      updateFileContent(activeFile, value);
    }
  };

  // Handle language change in editor
  const handleLanguageChange = (language) => {
    if (activeFile) {
      updateFileLanguage(activeFile, language);
    }
  };

  // Handle save action
  const handleSave = async () => {
    if (!selectedCodebase) {
      showToastMessage('Please select a codebase to save.', 'error');
      return;
    }

    try {
      const result = await saveChanges(selectedCodebase.cid);
      if (result.success) {
        showToastMessage('Changes saved successfully!', 'success');
      } else {
        showToastMessage(result.message || 'Failed to save changes.', 'error');
      }
    } catch (error) {
      showToastMessage('An error occurred while saving.', 'error');
    }
  };

  // Handle codebase create
  const handleCodebaseCreate = async (requestData) => {
    try {
      await handleCreateCodebase(requestData);
      showToastMessage('Codebase created successfully!', 'success');
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

  // Toast 메시지 표시 헬퍼
  const showToastMessage = (message, type = 'success') => {
    setToastMessage(message);
    setToastType(type);
    setShowToast(true);
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
              ) : editorLoading ? (
                <CodeEditorSkeleton compact={false} />
              ) : isEmpty ? (
                <div className={styles.emptyState}>
                  <h3>No Code Files</h3>
                  <p>The selected codebase contains no files.</p>
                </div>
              ) : (
                <CodeEditor
                  fileStructure={fileStructure}
                  files={files}
                  activeFile={activeFile}
                  onFileChange={changeActiveFile}
                  onFilesChange={() => {}} // Not used
                  onSaveSnapshot={handleSave}
                  snapshotName={selectedCodebase.name || selectedCodebase.cid}
                  compact={false}
                  hideSaveButtons={true} // Handled by top toolbar
                  currentFile={currentFile}
                  onEditorChange={handleEditorChange}
                  onLanguageChange={handleLanguageChange}
                  onSnapshotSave={handleSave}
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
