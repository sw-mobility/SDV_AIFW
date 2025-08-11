import React, { useState, useRef, useEffect, useMemo } from 'react';
import Editor from '@monaco-editor/react';
import { FixedSizeList as List } from 'react-window';
import { ChevronRight, ChevronDown, Save, X } from 'lucide-react';
import styles from './CodeEditor.module.css';
import Button from '../atoms/Button.jsx';
import IconifyIcon from '../atoms/IconifyIcon.jsx';
import { getFileIcon, getFolderIcon, getOpenFolderIcon } from '../../../utils/fileIcons.js';

/**
 * 가상화된 파일 트리 컴포넌트
 * 대용량 파일 구조도 부드럽게 렌더링
 */
const VirtualizedFileTree = ({ fileStructure, onFileClick, activeFile, height = 400 }) => {
    const [expandedFolders, setExpandedFolders] = useState(new Set(['src']));

    // 파일 트리를 평면 목록으로 변환
    const flattenedItems = useMemo(() => {
        const items = [];
        
        const traverse = (item, level = 0, parentPath = '') => {
            const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
            const isFolder = item.type === 'folder' || item.type === 'directory';
            const isExpanded = expandedFolders.has(currentPath);
            
            items.push({
                ...item,
                level,
                path: currentPath,
                isFolder,
                isExpanded
            });
            
            // 폴더가 열려있고 자식이 있으면 재귀적으로 추가
            if (isFolder && isExpanded && item.children) {
                item.children.forEach(child => {
                    traverse(child, level + 1, currentPath);
                });
            }
        };
        
        fileStructure.forEach(item => traverse(item));
        return items;
    }, [fileStructure, expandedFolders]);

    const toggleFolder = (path) => {
        setExpandedFolders(prev => {
            const newSet = new Set(prev);
            if (newSet.has(path)) {
                newSet.delete(path);
            } else {
                newSet.add(path);
            }
            return newSet;
        });
    };

    const FileTreeItem = ({ index, style }) => {
        const item = flattenedItems[index];
        if (!item) return null;

        const { name, level, path, isFolder, isExpanded } = item;
        const isActive = !isFolder && activeFile === path;

        return (
            <div style={style}>
                <div
                    className={
                        styles['file-item'] +
                        (isFolder ? ' ' + styles.folder : ' ' + styles.file) +
                        (isActive ? ' ' + styles.activeFile : '')
                    }
                    style={{ 
                        paddingLeft: `${level * 16 + 8}px`, 
                        cursor: 'pointer',
                        height: '24px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px'
                    }}
                    onClick={() => {
                        if (isFolder) {
                            toggleFolder(path);
                        } else {
                            onFileClick(path);
                        }
                    }}
                    title={path}
                >
                    {isFolder ? (
                        <>
                            {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                            <IconifyIcon 
                                icon={isExpanded ? getOpenFolderIcon(name) : getFolderIcon(name)} 
                                size={16} 
                            />
                        </>
                    ) : (
                        <IconifyIcon 
                            icon={getFileIcon(name)} 
                            size={16}
                        />
                    )}
                    <span className={styles['file-name']}>{name}</span>
                </div>
            </div>
        );
    };

    return (
        <List
            height={height}
            itemCount={flattenedItems.length}
            itemSize={24}
            className={styles['virtualized-file-tree']}
        >
            {FileTreeItem}
        </List>
    );
};

/**
 CodeEditor component
 Monaco Editor + file item
 Props:
 - fileStructure: 파일 트리 구조
 - files: { [filename]: { code, language } }
 - activeFile: 현재 열려 있는 파일명
 - onFileChange: (filename) => void
 - onFilesChange: (newFiles) => void
 - onSaveSnapshot: () => void
*/
function CodeEditor({
    fileStructure: propFileStructure,
    files: propFiles,
    activeFile: propActiveFile,
    onFileChange,
    onFilesChange,
    onSaveSnapshot,
    snapshotName = 'Default Snapshot',
    onCloseDrawer,
    compact = false,
    hideSaveButtons = false,
    currentFile,
    onEditorChange,
    onLanguageChange,
    onSnapshotSave,
}) {
    // Use props directly instead of internal state for fileStructure and files
    const fileStructure = propFileStructure || [
        {
            name: 'src',
            type: 'folder',
            children: [
                { name: 'train.py', type: 'file' },
                { name: 'data_loader.py', type: 'file' },
                { name: 'model_config.py', type: 'file' },
                { name: 'train_parameter.json', type: 'file' },
            ]
        }
    ];
    const files = propFiles || {
        'train.py': { code: `# Training script\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n    \n    def forward(self, x):\n        return self.linear(x)\n\nmodel = Model()\nprint("Model initialized successfully!")`, language: 'python' }
    };
    const activeFile = propActiveFile || (fileStructure[0]?.children?.[0]?.name || '');

    // 파일 클릭 핸들러
    const handleFileClick = (filename) => {
        if (onFileChange) onFileChange(filename);
    };

    // 코드 변경 핸들러
    const handleEditorChange = (value) => {
        if (onEditorChange) {
            onEditorChange(value);
        } else if (!activeFile) return;
        // Fallback for backward compatibility
        if (onFilesChange) {
            const newFiles = {
                ...files,
                [activeFile]: {
                    ...files[activeFile],
                    code: value
                }
            };
            onFilesChange(newFiles);
        }
    };

    // 언어 변경 핸들러
    const handleLanguageChange = (e) => {
        if (onLanguageChange) {
            onLanguageChange(e.target.value);
        } else if (!activeFile) return;
        // Fallback for backward compatibility
        if (onFilesChange) {
            const newLang = e.target.value;
            const newFiles = {
                ...files,
                [activeFile]: {
                    ...files[activeFile],
                    language: newLang
                }
            };
            onFilesChange(newFiles);
        }
    };

    const currentFileData = currentFile || files[activeFile] || { code: '', language: 'python' };

    return (
        <div className={`${styles['code-editor-container']} ${compact ? styles.compact : ''}`}>
            {!compact && (
            <div className={styles.sidebar}>
                <div className={styles['file-explorer']}>
                    <div className={styles.snapshotHeader}>
                        <div className={styles.snapshotName}>{snapshotName}</div>
                        {snapshotName === 'Default Snapshot' && <span className={styles.snapshotDefault}>(default)</span>}
                    </div>
                    <div className={styles['file-tree']}>
                        <VirtualizedFileTree
                            fileStructure={fileStructure}
                            onFileClick={handleFileClick}
                            activeFile={activeFile}
                            height={Math.max(300, window.innerHeight - 400)}
                        />
                    </div>
                </div>
                {/* Snapshot Save Button */}
                {!hideSaveButtons && (
                <div className={styles.snapshotSidebarBtnsWrap}>
                    <Button
                        onClick={onSnapshotSave}
                        variant="primary"
                        className={styles.snapshotSaveBtn}
                        disabled={!activeFile}
                        icon={<Save size={16} />}
                    >
                        Save Snapshot
                    </Button>
                </div>
                )}
            </div>
            )}

            <div className={styles['editor-main']}>
                {!compact && (
                <div className={styles['editor-toolbar']}>
                    <div className={styles['toolbar-left']}>
                        <div className={styles.fileInfo}>
                            <IconifyIcon 
                                icon={getFileIcon(activeFile)} 
                                size={16}
                            />
                            <span className={styles.activeFileName}>{activeFile}</span>
                        </div>
                        <select value={currentFileData.language} onChange={handleLanguageChange} className={styles['language-select']}>
                            <option value="json">JSON</option>
                            <option value="python">Python</option>
                            <option value="java">Java</option>
                            <option value="c">C</option>
                            <option value="markdown">Markdown</option>
                        </select>
                    </div>
                </div>
                )}
                <Editor
                    height="100%"
                    language={currentFileData.language}
                    value={currentFileData.code}
                    onChange={handleEditorChange}
                    theme="vs-light"
                    options={{
                        minimap: { enabled: !compact },
                        fontSize: compact ? 14 : 14,
                        wordWrap: 'on',
                        automaticLayout: true,
                        scrollBeyondLastLine: false,
                        folding: true,
                        lineNumbers: 'on',
                        bracketPairColorization: { enabled: true },
                        formatOnPaste: true,
                        formatOnType: true,
                        suggestOnTriggerCharacters: true,
                        quickSuggestions: true,
                        parameterHints: { enabled: true },
                        autoIndent: 'full',
                        tabSize: 2,
                        insertSpaces: true,
                        detectIndentation: true,
                        largeFileOptimizations: true,
                        contextmenu: true,
                        mouseWheelZoom: true,
                        smoothScrolling: true,
                        cursorBlinking: 'smooth',
                        cursorSmoothCaretAnimation: 'on',
                        renderWhitespace: 'selection',
                        guides: {
                            indentation: true,
                            bracketPairs: true
                        }
                    }}
                />
            </div>
        </div>
    );
}

export default CodeEditor;
