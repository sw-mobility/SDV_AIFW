import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Folder, File, ChevronRight, ChevronDown, PlusCircle, Save, X, Code } from 'lucide-react';
import styles from './CodeEditor.module.css';
import Button from './Button.jsx';

// Helper: get extension color
const extColor = ext => {
  if (ext === 'py') return '#4f8cff';
  if (ext === 'json') return '#ffb300';
  return '#bbb';
};

/**
 FileTree component
 @param {Object} item - file or folder (name, type, children)
 @param {number} level - 현재 depth level (들여쓰기 계산용)
 @param {function} onFileClick - 파일 클릭 핸들러
 @param {string} activeFile - 현재 열려 있는 파일명
 @param {boolean} defaultOpen - 기본적으로 열려있을지 여부
 */
const FileTree = ({ item, level = 0, onFileClick, activeFile, defaultOpen = false }) => {
    const [isOpened, setIsOpened] = useState(defaultOpen);
    const isFolder = item.type === 'folder';
    const ext = !isFolder && item.name.split('.').pop();
    
    return (
        <div>
            <div
                className={
                  styles['file-item'] +
                  (isFolder ? ' ' + styles.folder : ' ' + styles.file) +
                  (!isFolder && activeFile === item.name ? ' ' + styles.activeFile : '')
                }
                style={{ paddingLeft: `${level * 16 + 8}px`, cursor: 'pointer' }}
                onClick={() => {
                    if (isFolder) setIsOpened(!isOpened);
                    else onFileClick(item.name);
                }}
                title={item.name}
            >
                {isFolder ? (
                    <>
                        {isOpened ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                        <Folder size={14} />
                    </>
                ) : (
                    <>
                        <span className={styles.fileExtDot} style={{ background: extColor(ext) }}></span>
                        <File size={14} />
                    </>
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
}) {
    const [fileStructure, setFileStructure] = useState(propFileStructure || [
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
    ]);
    const [files, setFiles] = useState(propFiles || {
        'train.py': { code: `# Training script\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n    \n    def forward(self, x):\n        return self.linear(x)\n\nmodel = Model()\nprint("Model initialized successfully!")`, language: 'python' }
    });
    const [activeFile, setActiveFile] = useState(propActiveFile || fileStructure[0]?.children?.[0]?.name || '');
    const [snapshotNameInput, setSnapshotNameInput] = useState('');
    const [showNameInput, setShowNameInput] = useState(false);

    // 파일 클릭 핸들러
    const handleFileClick = (filename) => {
        setActiveFile(filename);
        if (onFileChange) onFileChange(filename);
    };

    // 코드 변경 핸들러
    const handleEditorChange = (value) => {
        if (!activeFile) return;
        const newFiles = {
            ...files,
            [activeFile]: {
                ...files[activeFile],
                code: value
            }
        };
        setFiles(newFiles);
        if (onFilesChange) onFilesChange(newFiles);
    };

    // 언어 변경 핸들러
    const handleLanguageChange = (e) => {
        if (!activeFile) return;
        const newLang = e.target.value;
        const newFiles = {
            ...files,
            [activeFile]: {
                ...files[activeFile],
                language: newLang
            }
        };
        setFiles(newFiles);
        if (onFilesChange) onFilesChange(newFiles);
    };

    // 현재 파일 정보
    const currentFile = files[activeFile] || { code: '', language: 'python' };

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
                            <Code size={14} />
                            <span className={styles.activeFileName}>{activeFile}</span>
                        </div>
                        <select value={currentFile.language} onChange={handleLanguageChange} className={styles['language-select']}>
                            <option value="json">JSON</option>
                            <option value="python">Python</option>
                            <option value="javascript">JavaScript</option>
                            <option value="typescript">TypeScript</option>
                            <option value="java">Java</option>
                            <option value="cpp">C++</option>
                            <option value="html">HTML</option>
                            <option value="css">CSS</option>
                            <option value="markdown">Markdown</option>
                        </select>
                    </div>
                </div>
                )}
                <Editor
                    height="100%"
                    language={currentFile.language}
                    value={currentFile.code}
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
