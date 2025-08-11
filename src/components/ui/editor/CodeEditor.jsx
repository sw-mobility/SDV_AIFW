import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { ChevronRight, ChevronDown, PlusCircle, Save, X } from 'lucide-react';
import styles from './CodeEditor.module.css';
import Button from '../atoms/Button.jsx';
import IconifyIcon from '../atoms/IconifyIcon.jsx';
import { getFileIcon, getFolderIcon, getOpenFolderIcon } from '../../../utils/fileIcons.js';

/**
 FileTree component
 @param {Object} item - file or folder (name, type, children)
 @param {number} level - 현재 depth level (들여쓰기 계산용)
 @param {function} onFileClick - 파일 클릭 핸들러
 @param {string} activeFile - 현재 열려 있는 파일명
 @param {boolean} defaultOpen - 기본적으로 열려있을지 여부
 */
const FileTree = ({ item, level = 0, onFileClick, activeFile, defaultOpen = false, parentPath = '' }) => {
    const [isOpened, setIsOpened] = useState(defaultOpen);
    const isFolder = item.type === 'folder' || item.type === 'directory';
    
    // 현재 아이템의 전체 경로 계산
    const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
    const isActive = !isFolder && activeFile === currentPath;
    
    return (
        <div>
            <div
                className={
                  styles['file-item'] +
                  (isFolder ? ' ' + styles.folder : ' ' + styles.file) +
                  (isActive ? ' ' + styles.activeFile : '')
                }
                style={{ paddingLeft: `${level * 16 + 8}px`, cursor: 'pointer' }}
                onClick={() => {
                    if (isFolder) setIsOpened(!isOpened);
                    else onFileClick(currentPath);
                }}
                title={currentPath}
            >
                {isFolder ? (
                    <>
                        {isOpened ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                        <IconifyIcon 
                            icon={isOpened ? getOpenFolderIcon(item.name) : getFolderIcon(item.name)} 
                            size={16} 
                        />
                    </>
                ) : (
                    <IconifyIcon 
                        icon={getFileIcon(item.name)} 
                        size={16}
                    />
                )}
                <span className={styles['file-name']}>{item.name}</span>
            </div>
            {isFolder && isOpened && item.children && (
                <div className={styles['file-children']}>
                    {item.children.map((child, index) => (
                        <FileTree 
                            key={index} 
                            item={child} 
                            level={level + 1} 
                            onFileClick={onFileClick} 
                            activeFile={activeFile}
                            parentPath={currentPath}
                        />
                    ))}
                </div>
            )}
        </div>
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
    const [snapshotNameInput, setSnapshotNameInput] = useState('');
    const [showNameInput, setShowNameInput] = useState(false);

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
                        {fileStructure.map((item, index) => (
                            <FileTree 
                                key={index} 
                                item={item} 
                                onFileClick={handleFileClick} 
                                activeFile={activeFile} 
                                defaultOpen={item.name === 'src'} 
                            />
                        ))}
                    </div>
                </div>
                {/* Snapshot Save Buttons */}
                {!hideSaveButtons && (
                <div className={styles.snapshotSidebarBtnsWrap}>
                    {showNameInput && (
                        <div className={styles.snapshotInputWrapper}>
                            <input
                                type="text"
                                className={styles.snapshotInput}
                                value={snapshotNameInput}
                                onChange={e => setSnapshotNameInput(e.target.value)}
                                placeholder="Enter new snapshot name"
                                autoFocus
                            />
                            <button
                                className={styles.snapshotBtn + ' ' + styles.cancel}
                                onClick={() => { setShowNameInput(false); setSnapshotNameInput(''); }}
                                title="Cancel"
                            >
                                <X size={16} />
                            </button>
                        </div>
                    )}
                    <div className={styles.snapshotModalBtns}>
                        <button
                            className={styles.snapshotBtn}
                            onClick={() => {
                                if (!showNameInput) {
                                    setShowNameInput(true);
                                } else if (snapshotNameInput.trim()) {
                                    onSaveSnapshot(snapshotNameInput.trim());
                                    if (onCloseDrawer) onCloseDrawer();
                                    setShowNameInput(false);
                                    setSnapshotNameInput('');
                                }
                            }}
                            disabled={showNameInput && !snapshotNameInput.trim()}
                        >
                            <PlusCircle size={16} /> Save as New Snapshot
                        </button>
                        <button
                            className={styles.snapshotBtn + ' ' + styles.overwrite}
                            onClick={() => {
                                if (snapshotName !== 'Default Snapshot') {
                                    onSaveSnapshot(snapshotName);
                                    if (onCloseDrawer) onCloseDrawer();
                                    setShowNameInput(false);
                                    setSnapshotNameInput('');
                                }
                            }}
                            disabled={snapshotName === 'Default Snapshot'}
                        >
                            <Save size={16} /> Overwrite Current Snapshot
                        </button>
                    </div>
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
