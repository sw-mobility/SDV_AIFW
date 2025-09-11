import React from 'react';
import { SkeletonText } from '../atoms/Skeleton.jsx';
import IconifyIcon from '../atoms/IconifyIcon.jsx';
import styles from './CodeEditor.module.css';
import skeletonStyles from './CodeEditorSkeleton.module.css';

/**
 * 파일 트리 스켈레톤 아이템
 */
const FileTreeSkeletonItem = ({ level = 0, isFolder = false, fileName = 'default' }) => {
  const getSkeletonIcon = () => {
    if (isFolder) {
      return 'ph:folder';
    }

    const fileTypes = [
      'vscode-icons:file-type-python',
      'vscode-icons:file-type-python',
      'vscode-icons:file-type-yaml',
      'vscode-icons:file-type-json',
      'vscode-icons:file-type-markdown',
      'vscode-icons:default-file'
    ];
    
    return fileTypes[level % fileTypes.length];
  };

  return (
    <div className={styles['file-item']} style={{ paddingLeft: `${level * 16 + 8}px` }}>
      <IconifyIcon 
        icon={getSkeletonIcon()} 
        size={16} 
      />
      <div className={skeletonStyles.fileNameSkeleton}>
        <SkeletonText 
          lines={1} 
          width={isFolder ? "80px" : "120px"} 
          height="14px"
        />
      </div>
    </div>
  );
};

/**
 * 파일 트리 스켈레톤
 */
const FileTreeSkeleton = () => {
  return (
    <div className={styles['file-explorer']}>
      {/* Snapshot Header Skeleton */}
      <div className={styles.snapshotHeader}>
        <div className={skeletonStyles.snapshotNameSkeleton}>
          <SkeletonText lines={1} width="150px" height="16px" />
        </div>
      </div>
      
      {/* File Tree Skeleton */}
      <div className={styles['file-tree']}>
        {/* Root folder */}
        <FileTreeSkeletonItem level={0} isFolder={true} />
        
        {/* Sub items */}
        <FileTreeSkeletonItem level={1} isFolder={false} />
        <FileTreeSkeletonItem level={1} isFolder={false} />
        <FileTreeSkeletonItem level={1} isFolder={true} />
        <FileTreeSkeletonItem level={2} isFolder={false} />
        <FileTreeSkeletonItem level={2} isFolder={false} />
        <FileTreeSkeletonItem level={1} isFolder={false} />
        <FileTreeSkeletonItem level={1} isFolder={true} />
        <FileTreeSkeletonItem level={2} isFolder={false} />
        <FileTreeSkeletonItem level={2} isFolder={false} />
        <FileTreeSkeletonItem level={2} isFolder={false} />
        <FileTreeSkeletonItem level={1} isFolder={false} />
      </div>
    </div>
  );
};

/**
 * 에디터 툴바 스켈레톤
 */
const EditorToolbarSkeleton = () => {
  return (
    <div className={styles['editor-toolbar']}>
      <div className={styles['toolbar-left']}>
        <div className={styles.fileInfo}>
          <IconifyIcon 
            icon="vscode-icons:file-type-python" 
            size={16} 
          />
          <div className={skeletonStyles.fileNameSkeleton}>
            <SkeletonText lines={1} width="100px" height="14px" />
          </div>
        </div>
        <div className={skeletonStyles.languageSelectSkeleton}>
          <SkeletonText lines={1} width="80px" height="32px" />
        </div>
      </div>
    </div>
  );
};

/**
 * 에디터 코드 영역 스켈레톤
 */
const EditorCodeSkeleton = () => {
  return (
    <div className={skeletonStyles.editorCodeSkeleton}>
      {/* Line numbers skeleton */}
      <div className={skeletonStyles.lineNumbers}>
        {Array(20).fill(null).map((_, index) => (
          <div key={index} className={skeletonStyles.lineNumber}>
            <SkeletonText lines={1} width="20px" height="16px" />
          </div>
        ))}
      </div>
      
      {/* Code content skeleton */}
      <div className={skeletonStyles.codeContent}>
        {Array(20).fill(null).map((_, index) => (
          <div key={index} className={skeletonStyles.codeLine}>
            <SkeletonText 
              lines={1} 
              width={`${Math.random() * 60 + 20}%`} 
              height="16px" 
            />
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * 코드 에디터 전체 스켈레톤
 */
const CodeEditorSkeleton = ({ compact = false }) => {
  return (
    <div className={`${styles['code-editor-container']} ${compact ? styles.compact : ''}`}>
      {!compact && (
        <div className={styles.sidebar}>
          <FileTreeSkeleton />
        </div>
      )}
      
      <div className={styles['editor-main']}>
        {!compact && <EditorToolbarSkeleton />}
        <EditorCodeSkeleton />
      </div>
    </div>
  );
};

export default CodeEditorSkeleton;
