import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
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
    const listRef = useRef(null);
    const [scrollOffset, setScrollOffset] = useState(0);

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

    const isFolderItem = (item) => item.type === 'folder' || item.type === 'directory';

    const findNodeByPath = (items, targetPath, parentPath = '') => {
        for (const item of items) {
            const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
            if (currentPath === targetPath) return { node: item, parentPath };
            if (isFolderItem(item) && item.children) {
                const found = findNodeByPath(item.children, targetPath, currentPath);
                if (found) return found;
            }
        }
        return null;
    };

    const countVisibleDescendants = (node, nodePath, expandedSet) => {
        if (!node || !isFolderItem(node) || !node.children) return 0;
        let count = 0;
        for (const child of node.children) {
            count += 1;
            const childPath = `${nodePath}/${child.name}`;
            if (isFolderItem(child) && expandedSet.has(childPath)) {
                count += countVisibleDescendants(child, childPath, expandedSet);
            }
        }
        return count;
    };

    const ITEM_SIZE = 24;

    const toggleFolder = (path) => {
        setExpandedFolders(prev => {
            const toggledIndex = flattenedItems.findIndex(i => i.path === path);
            const firstVisibleIndex = Math.floor(scrollOffset / ITEM_SIZE);
            const found = findNodeByPath(fileStructure, path);

            const newSet = new Set(prev);
            const isExpanding = !newSet.has(path);
            let deltaItems = 0;

            if (isExpanding) {
                newSet.add(path);
                if (found) {
                    deltaItems = countVisibleDescendants(found.node, path, newSet);
                }
            } else {
                // collapsing
                if (found) {
                    deltaItems = -countVisibleDescendants(found.node, path, prev);
                }
                newSet.delete(path);
            }

            // If the toggled folder is above the viewport, compensate scroll to anchor content
            if (listRef.current && toggledIndex !== -1 && toggledIndex <= firstVisibleIndex && deltaItems !== 0) {
                listRef.current.scrollTo(scrollOffset + deltaItems * ITEM_SIZE);
            }

            return newSet;
        });
    };

    const FileTreeRow = React.memo(({ index, style, data }) => {
        const item = data.items[index];
        if (!item) return null;

        const { name, level, path, isFolder, isExpanded } = item;
        const isActive = !isFolder && data.activeFile === path;

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
                            data.toggleFolder(path);
                        } else {
                            data.onFileClick(path);
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
    }, (prev, next) => {
        // Avoid re-render if index and style are the same and this row is not affected by activeFile change
        if (prev.index !== next.index) return false;
        if (prev.style.top !== next.style.top || prev.style.height !== next.style.height) return false;
        const prevItems = prev.data.items;
        const nextItems = next.data.items;
        if (prevItems !== nextItems) return false; // structure/expand changed
        const prevActive = prev.data.activeFile;
        const nextActive = next.data.activeFile;
        if (prevActive !== nextActive) {
            const item = prevItems[prev.index];
            const path = item?.path;
            if (path === prevActive || path === nextActive) return false; // this row toggles active style
            return true; // skip rerender for unaffected rows
        }
        return true;
    });

    const itemData = useMemo(() => ({
        items: flattenedItems,
        activeFile,
        onFileClick,
        toggleFolder
    }), [flattenedItems, activeFile, onFileClick, toggleFolder]);

    return (
        <List
            ref={listRef}
            height={height}
            width={"100%"}
            itemCount={flattenedItems.length}
            itemSize={ITEM_SIZE}
            overscanCount={8}
            onScroll={({ scrollOffset: so }) => setScrollOffset(so)}
            itemKey={(index) => flattenedItems[index]?.path || index}
            itemData={itemData}
            className={styles['virtualized-file-tree']}
        >
            {FileTreeRow}
        </List>
    );
};

// Avoid unnecessary re-renders when unrelated props change
const MemoizedVirtualizedFileTree = React.memo(VirtualizedFileTree);

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
export default function CodeEditor({
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
    onEditorMount, // 에디터 인스턴스를 부모로 전달하는 콜백
}) {
    // Monaco Editor 인스턴스 참조
    const editorRef = useRef(null);
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

    // 동적 사이드바 파일 트리 높이 계산 (윈도우 리사이즈 대응)
    const [fileTreeHeight, setFileTreeHeight] = useState(
        typeof window !== 'undefined' ? Math.max(300, window.innerHeight - 400) : 400
    );
    useEffect(() => {
        const handleResize = () => {
            setFileTreeHeight(Math.max(300, window.innerHeight - 400));
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // 파일 클릭 핸들러 (stable reference)
    const handleFileClick = useCallback((filename) => {
        if (onFileChange) onFileChange(filename);
    }, [onFileChange]);

    // Monaco Editor 마운트 핸들러
    const handleEditorDidMount = useCallback((editor, monaco) => {
        editorRef.current = editor;
        
        // 강화된 이벤트 리스너 추가
        editor.onDidChangeModelContent(() => {
            const value = editor.getValue();
            if (onEditorChange) {
                onEditorChange(value);
            }
        });
        
        // 부모 컴포넌트에 에디터 인스턴스 전달
        if (onEditorMount) {
            onEditorMount(editor);
        }
    }, [onEditorMount, onEditorChange]);

    // 코드 변경 핸들러
    const handleEditorChange = useCallback((value) => {
        if (onEditorChange) {
            onEditorChange(value);
        } else if (activeFile && onFilesChange) {
            // Fallback for backward compatibility
            const newFiles = {
                ...files,
                [activeFile]: {
                    ...files[activeFile],
                    code: value
                }
            };
            onFilesChange(newFiles);
        }
    }, [activeFile, onEditorChange, onFilesChange, files]);

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
                        <MemoizedVirtualizedFileTree
                            fileStructure={fileStructure}
                            onFileClick={handleFileClick}
                            activeFile={activeFile}
                            height={fileTreeHeight}
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
                            <option value="yaml">YAML</option>
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
                    onMount={handleEditorDidMount}
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
