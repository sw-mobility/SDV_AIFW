import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import { Folder, File, ChevronRight, ChevronDown } from 'lucide-react';
import styles from './CodeEditor.module.css';

/**
 FileTree component

 @param {Object} item - file or folder (name, type, children)
 @param {number} level - 현재 depth level (들여쓰기 계산용)
 */
const FileTree = ({ item, level = 0 }) => {
    const [isOpened, setIsOpened] = useState(false);
    const isFolder = item.type === 'folder';

    return (
        <div>
            <div
                className={`${styles['file-item']} ${isFolder ? styles.folder : styles.file}`}
                style={{ paddingLeft: `${level * 16 + 8}px` }} // 들여쓰기
                onClick={() => isFolder && setIsOpened(!isOpened)}
            >
                {isFolder ? (
                    <>
                        {isOpened ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                        <Folder size={12} />
                    </>
                ) : (
                    <File size={12} />
                )}
                <span className={styles['file-name']}>{item.name}</span>
            </div>

            {isFolder && isOpened && item.children && (
                <div className={styles['file-children']}>
                    {item.children.map((child, index) => (
                        <FileTree key={index} item={child} level={level + 1} />
                    ))}
                </div>
            )}
        </div>
    );
};

/**
 CodeEditor component
 Monaco Editor + file item
 */
function CodeEditor() {
    const [language, setLanguage] = useState('python');
    const [code, setCode] = useState(`#python\nprint('Right triangle\\n')\nleg = int(raw_input('leg: '))`);

    // 파일 트리 구조 (example)
    const fileStructure = [
        {
            name: 'src',
            type: 'folder',
            children: [
                { name: 'data_loader.py', type: 'file' },
                { name: 'train.py', type: 'file' },
                { name: 'model_config.py', type: 'file' },
                { name: 'train_parameter.json', type: 'file' },
            ]
        }
    ];

    /**
     Monaco Editor handler
     @param {string} value - 수정 후 전체 코드
     @param {object} event - 이번 변경에서 수정된 부분
     */
    const handleEditorChange = (value, event) => {
        setCode(value);
        console.log('Current code:', value);
    };

    const handleLanguageChange = (e) => {
        setLanguage(e.target.value);
    };

    const handleSaveCode = () => {

    };

    return (
        <div className={styles['code-editor-container']}>
            <div className={styles.sidebar}>
                <div className={styles['file-explorer']}>
                    <h3>project name</h3>
                    <div className={styles['file-tree']}>
                        {fileStructure.map((item, index) => (
                            <FileTree key={index} item={item} />
                        ))}
                    </div>
                </div>
            </div>

            <div className={styles['editor-main']}>
                <div className={styles['editor-toolbar']}>
                    <div className={styles['toolbar-left']}>
                        <select value={language} onChange={handleLanguageChange} className={styles['language-select']}>
                            <option value="json">JSON</option>
                            <option value="python">Python</option>
                            <option value="java">Java</option>
                            <option value="cpp">C++</option>
                        </select>
                    </div>
                </div>
                <Editor
                    height="calc(100vh - 120px)"
                    language={language}
                    value={code}
                    onChange={handleEditorChange}
                    theme="vs-light"
                    options={{
                        minimap: { enabled: true },
                        fontSize: 14,
                        wordWrap: 'on', //긴 줄 자동 줄바꿈
                        automaticLayout: true,//parent container 크기 변경 시 자동 resize
                        scrollBeyondLastLine: false,
                        folding: true,//코드 폴딩
                        lineNumbers: 'on',
                        bracketPairColorization: { enabled: true },//괄호 쌍 색상 구분
                        formatOnPaste: true,
                        formatOnType: true, //; 입력 자동 정렬
                        suggestOnTriggerCharacters: true,
                        quickSuggestions: true,//자동완성 제안
                        parameterHints: { enabled: true },//함수 호출 파라미터 힌트
                        autoIndent: 'full',
                        tabSize: 2,
                        insertSpaces: true,
                        detectIndentation: true,
                        largeFileOptimizations: true,//큰 파일 편집 시 최적화
                        contextmenu: true,//우클릭 메뉴
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
