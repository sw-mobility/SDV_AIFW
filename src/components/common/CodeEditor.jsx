import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Folder, File, ChevronRight, ChevronDown, PlusCircle, Save, X } from 'lucide-react';
import styles from './CodeEditor.module.css';
import Button from './Button.jsx';

/**
 FileTree component
 @param {Object} item - file or folder (name, type, children)
 @param {number} level - 현재 depth level (들여쓰기 계산용)
 @param {function} onFileClick - 파일 클릭 핸들러
 @param {string} activeFile - 현재 열려 있는 파일명
 */
const FileTree = ({ item, level = 0, onFileClick, activeFile }) => {
    const [isOpened, setIsOpened] = useState(false);
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
                        {isOpened ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                        <Folder size={12} />
                    </>
                ) : (
                    <>
                        <span className={styles.fileExtDot} style={{ background: extColor(ext) }}></span>
                        <File size={12} />
                    </>
                )}
                <span className={styles['file-name']} style={{ maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</span>
            </div>
            {isFolder && isOpened && item.children && (
                <div className={styles['file-children']}>
                    {item.children.map((child, index) => (
                        <FileTree key={index} item={child} level={level + 1} onFileClick={onFileClick} activeFile={activeFile} />
                    ))}
                </div>
            )}
        </div>
    );
};

// Helper: get extension color
const extColor = ext => {
  if (ext === 'py') return '#4f8cff';
  if (ext === 'json') return '#ffb300';
  if (ext === 'js') return '#f7df1e';
  if (ext === 'cpp') return '#b0bec5';
  if (ext === 'java') return '#e76f00';
  return '#bbb';
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
}) {
    // 내부 상태: 열려 있는 파일
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
        'train.py': { code: `#python\nprint('Right triangle\\n')\nleg = int(raw_input('leg: '))`, language: 'python' }
    });
    const [activeFile, setActiveFile] = useState(propActiveFile || fileStructure[0]?.children?.[0]?.name || '');
    const [snapshotNameInput, setSnapshotNameInput] = useState('');
    const [showNameInput, setShowNameInput] = useState(false);

    // FileTree (read-only, src 기본 오픈)
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
                            {isOpened ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                            <Folder size={12} />
                        </>
                    ) : (
                        <>
                            <span className={styles.fileExtDot} style={{ background: extColor(ext) }}></span>
                            <File size={12} />
                        </>
                    )}
                    <span className={styles['file-name']} style={{ maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name}</span>
                </div>
                {isFolder && isOpened && item.children && (
                    <div className={styles['file-children']}>
                        {item.children.map((child, index) => (
                            <FileTree key={index} item={child} level={level + 1} onFileClick={onFileClick} activeFile={activeFile} />
                        ))}
                    </div>
                )}
            </div>
        );
    };

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

    // 파일 트리 구조
    // const fileStructure = propFileStructure || [
    //     {
    //         name: 'src',
    //         type: 'folder',
    //         children: [
    //             { name: 'train.py', type: 'file' },
    //             { name: 'data_loader.py', type: 'file' },
    //             { name: 'model_config.py', type: 'file' },
    //             { name: 'train_parameter.json', type: 'file' },
    //         ]
    //     }
    // ];

    // 현재 파일 정보
    const currentFile = files[activeFile] || { code: '', language: 'python' };

    return (
        <div className={styles['code-editor-container']}>
            <div className={styles.sidebar}>
                <div className={styles['file-explorer']}>
                    <div style={{ marginBottom: 10 }}>
                        <div className={styles.snapshotName}>{snapshotName}</div>
                        {snapshotName === 'Default Snapshot' && <span className={styles.snapshotDefault}>(default)</span>}
                    </div>
                    <div className={styles['file-tree']} style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
                        {fileStructure.map((item, index) => (
                            <FileTree key={index} item={item} onFileClick={handleFileClick} activeFile={activeFile} defaultOpen={item.name === 'src'} />
                        ))}
                    </div>
                </div>
                {/* Snapshot Save Buttons */}
                <div className={styles.snapshotSidebarBtnsWrap}>
                    {showNameInput && (
                        <div style={{ width: '100%', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                            <input
                                type="text"
                                className={styles.snapshotInput}
                                value={snapshotNameInput}
                                onChange={e => setSnapshotNameInput(e.target.value)}
                                placeholder="Enter new snapshot name"
                                autoFocus
                                style={{ flex: 1 }}
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
            </div>

            <div className={styles['editor-main']}>
                <div className={styles['editor-toolbar']}>
                    <div className={styles['toolbar-left']}>
                        <select value={currentFile.language} onChange={handleLanguageChange} className={styles['language-select']}>
                            <option value="json">JSON</option>
                            <option value="python">Python</option>
                            <option value="java">Java</option>
                            <option value="cpp">C++</option>
                        </select>
                    </div>
                </div>
                <Editor
                    height="100%"
                    language={currentFile.language}
                    value={currentFile.code}
                    onChange={handleEditorChange}
                    theme="vs-light"
                    options={{
                        minimap: { enabled: true },
                        fontSize: 14,
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
                        cursorSmoothCaretAnimation: 'on'
                    }}
                />
            </div>
        </div>
    );
}

export default CodeEditor;
