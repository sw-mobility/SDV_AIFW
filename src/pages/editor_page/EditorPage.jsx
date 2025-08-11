import React, { useState } from 'react';
import AlgorithmSelector from '../../components/features/training/AlgorithmSelector.jsx';
import CodeEditor from '../../components/ui/editor/CodeEditor.jsx';
import CodeEditorSkeleton from '../../components/ui/editor/CodeEditorSkeleton.jsx';
import { useTrainingCore } from '../../hooks/training/useTrainingCore.js';
import { useCodeEditor } from '../../hooks/editor/useCodeEditor.js';
import Button from '../../components/ui/atoms/Button.jsx';
import ErrorMessage from '../../components/ui/atoms/ErrorMessage.jsx';
import styles from './EditorPage.module.css';

const EditorPage = () => {
  const [showSaveSuccess, setShowSaveSuccess] = useState(false);
  
  // Core training state (algorithm selection)
  const { algorithm, setAlgorithm } = useTrainingCore();
  
  // Code editor state and functionality
  const {
    fileStructure,
    files,
    activeFile,
    loading,
    error,
    hasUnsavedChanges,
    lastSavedAt,
    currentFile,
    isEmpty,
    updateFileContent,
    updateFileLanguage,
    changeActiveFile,
    saveChanges,
    discardChanges,
    createNewFile
  } = useCodeEditor(algorithm);

  // Handle algorithm change
  const handleAlgorithmChange = (newAlgorithm) => {
    if (hasUnsavedChanges) {
      const confirmChange = window.confirm(
        'You have unsaved changes. Are you sure you want to switch algorithms? Your changes will be lost.'
      );
      if (!confirmChange) return;
    }
    setAlgorithm(newAlgorithm);
  };

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
    const result = await saveChanges();
    if (result.success) {
      setShowSaveSuccess(true);
      setTimeout(() => setShowSaveSuccess(false), 3000);
    }
  };

  // Handle file changes from CodeEditor
  const handleFilesChange = (newFiles) => {
    // This is called when files are modified in the CodeEditor
    // We're already handling this through updateFileContent
  };

  return (
    <div className={styles.pageContainer}>
      <div className={styles.container}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Code Editor</h1>
          <p className={styles.pageDescription}>
            Configure algorithm parameters and edit training code with our integrated development environment.
          </p>
        </div>

        <div className={styles.selectorSection}>
          <AlgorithmSelector
            algorithm={algorithm}
            onAlgorithmChange={handleAlgorithmChange}
          />
          
          {/* Status and Action Buttons */}
          <div className={styles.statusSection}>
            {error && <ErrorMessage message={error} />}
            {showSaveSuccess && (
              <div className={styles.successMessage}>
                ✅ Changes saved successfully!
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Code Editor Section - Full Height */}
      <div className={styles.editorSection}>
        <div className={styles.editorContainer}>
          {loading ? (
            <CodeEditorSkeleton compact={false} />
          ) : isEmpty ? (
            <div className={styles.emptyState}>
              <h3>No Code Template Available</h3>
              <p>Please select an algorithm to load the corresponding code template.</p>
            </div>
          ) : (
            <CodeEditor
              fileStructure={fileStructure}
              files={files}
              activeFile={activeFile}
              onFileChange={changeActiveFile}
              onFilesChange={handleFilesChange}
              onSaveSnapshot={handleSave}
              snapshotName={`${algorithm} Template`}
              compact={false}
              hideSaveButtons={false}
              // Custom props for our enhanced functionality
              currentFile={currentFile}
              onEditorChange={handleEditorChange}
              onLanguageChange={handleLanguageChange}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default EditorPage;
