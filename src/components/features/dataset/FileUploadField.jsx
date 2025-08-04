import React, { useRef, useState, useMemo } from 'react';
import styles from './FileUploadField.module.css';
import { Upload, ChevronDown, ChevronUp, Folder, File } from 'lucide-react';

/**
 * 파일 업로드를 위한 field
 * @param files
 * @param setFiles
 * @param fileError
 * @param setFileError
 * @param accept
 * @param multiple
 * @param maxSizeMB
 * @param allowFolders - 폴더 선택 허용 여부
 * @returns {Element}
 * @constructor
 */
export default function FileUploadField({ 
    files, 
    setFiles, 
    fileError, 
    setFileError, 
    accept = '*', 
    multiple = true, 
    maxSizeMB = 1000,
    allowFolders = true 
}) {
    const fileInputRef = useRef();
    const folderInputRef = useRef();
    const [dragActive, setDragActive] = useState(false);
    const [showAllFiles, setShowAllFiles] = useState(false);
    const [filesPerPage] = useState(50); // 한 번에 표시할 파일 수
    const [currentPage, setCurrentPage] = useState(0);
    const [uploadMode, setUploadMode] = useState('files'); // 'files' or 'folders'

    // 파일 검증 로직에서 maxFiles 제한 제거
    const validateFiles = (selected) => {
        // 파일 형식 제한 제거 - 모든 파일 허용
        const totalSize = [...files, ...selected].reduce((a, f) => a + f.size, 0);
        if (totalSize > maxSizeMB * 1024 * 1024) {
            setFileError(`전체 파일 용량이 ${maxSizeMB}MB를 초과할 수 없습니다.`);
            return false;
        }
        setFileError && setFileError(null);
        return true;
    };

    const handleFileChange = (e) => {
        let selected = Array.from(e.target.files);
        if (!multiple) selected = selected.slice(0, 1);
        
        if (!validateFiles(selected)) return;
        setFiles(prev => [...prev, ...selected]);
    };

    // drag and drop 지원
    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        let selected = Array.from(e.dataTransfer.files);
        if (!multiple) selected = selected.slice(0, 1);
        
        if (!validateFiles(selected)) return;
        setFiles(prev => [...prev, ...selected]);
    };
    
    const handleDragOver = (e) => {
        e.preventDefault();
        setDragActive(true);
    };
    
    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragActive(false);
    };

    const handleRemoveFile = (fileToRemove) => {
        const newFiles = files.filter(f => !(f.name === fileToRemove.name && f.size === fileToRemove.size));
        setFiles(newFiles);
        setFileError && setFileError(null);
    };

    const totalSizeMB = (files.reduce((a, f) => a + f.size, 0) / (1024 * 1024)).toFixed(1);

    // 파일 목록 페이지네이션
    const paginatedFiles = useMemo(() => {
        if (showAllFiles) {
            return files;
        }
        const startIndex = currentPage * filesPerPage;
        return files.slice(startIndex, startIndex + filesPerPage);
    }, [files, showAllFiles, currentPage, filesPerPage]);

    const totalPages = Math.ceil(files.length / filesPerPage);
    const hasMoreFiles = files.length > filesPerPage;

    const getFriendlyError = (err) => {
        if (!err) return null;
        if (typeof err === 'string' && (err.includes('duplicate key') || err.includes('E11000'))) {
            return 'Some files already exist in this dataset.';
        }
        if (typeof err === 'string' && err.includes('batch op errors occurred')) {
            return 'Some files could not be uploaded because they already exist.';
        }
        return err;
    };

    // 폴더별 파일 그룹화 (UI 표시용)
    const groupedFiles = useMemo(() => {
        const groups = {};
        files.forEach(file => {
            const path = file.webkitRelativePath || file.name;
            const folder = path.split('/').slice(0, -1).join('/') || 'Root';
            if (!groups[folder]) {
                groups[folder] = [];
            }
            groups[folder].push(file);
        });
        return groups;
    }, [files]);

    // 파일 표시명 가져오기
    const getFileDisplayName = (file) => {
        const path = file.webkitRelativePath || file.name;
        return path.split('/').pop();
    };

    // 파일 상대 경로 가져오기
    const getFileRelativePath = (file) => {
        return file.webkitRelativePath || file.name;
    };

    // 파일 폴더 경로 가져오기
    const getFileFolderPath = (file) => {
        const path = file.webkitRelativePath || file.name;
        return path.split('/').slice(0, -1).join('/');
    };

    return (
        <div
            className={[
                styles.uploadField,
                dragActive ? styles.dragActive : '',
                fileError ? styles.error : '',
            ].join(' ')}
            tabIndex={0}
            aria-label="파일 업로드 영역"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDragEnd={handleDragLeave}
        >
            <div className={styles.headerRow}>
                <span className={styles.iconWrap}><Upload size={20} /></span>
                <span className={styles.title}>Drag or click to upload files or folders</span>
                <span className={styles.info}>
                    (unlimited files, {maxSizeMB}MB total)
                </span>
            </div>

            {/* 업로드 모드 선택 및 파일/폴더 선택 버튼 */}
            <div className={styles.uploadControls}>
                <div className={styles.uploadModeSelector}>
                    <button
                        type="button"
                        className={`${styles.modeBtn} ${uploadMode === 'files' ? styles.activeMode : ''}`}
                        onClick={() => setUploadMode('files')}
                    >
                        <File size={14} />
                        Files
                    </button>
                    {allowFolders && (
                        <button
                            type="button"
                            className={`${styles.modeBtn} ${uploadMode === 'folders' ? styles.activeMode : ''}`}
                            onClick={() => setUploadMode('folders')}
                        >
                            <Folder size={14} />
                            Folders
                        </button>
                    )}
                </div>
                
                <button
                    type="button"
                    className={styles.selectBtn}
                    onClick={() => {
                        if (uploadMode === 'folders') {
                            folderInputRef.current && folderInputRef.current.click();
                        } else {
                            fileInputRef.current && fileInputRef.current.click();
                        }
                    }}
                >
                    {uploadMode === 'folders' ? (
                        <>
                            <Folder size={14} />
                            Select Folders
                        </>
                    ) : (
                        <>
                            <File size={14} />
                            Select Files
                        </>
                    )}
                </button>
            </div>
            
            {files && files.length > 0 ? (
                <div className={styles.fileListContainer}>
                    {/* 폴더별 그룹 표시 */}
                    {uploadMode === 'folders' && Object.keys(groupedFiles).length > 0 && (
                        <div className={styles.folderGroups}>
                            {Object.entries(groupedFiles).map(([folder, folderFiles]) => (
                                <div key={folder} className={styles.folderGroup}>
                                    <div className={styles.folderHeader}>
                                        <Folder size={14} />
                                        <span className={styles.folderName}>{folder}</span>
                                        <span className={styles.fileCount}>({folderFiles.length} files)</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className={styles.fileList}>
                        {paginatedFiles.map((f) => (
                            <span key={f.name + f.size} className={styles.fileTag}>
                                <span className={styles.fileName} title={getFileRelativePath(f)}>
                                    {getFileDisplayName(f)}
                                </span>
                                {getFileRelativePath(f) !== f.name && (
                                    <span className={styles.filePath} title={getFileRelativePath(f)}>
                                        ({getFileFolderPath(f)})
                                    </span>
                                )}
                                <span className={styles.fileSize}>({(f.size / (1024 * 1024)).toFixed(1)}MB)</span>
                                <button 
                                    type="button" 
                                    aria-label="파일 삭제" 
                                    className={styles.removeBtn} 
                                    onClick={e => { e.stopPropagation(); handleRemoveFile(f); }}
                                >
                                    ×
                                </button>
                            </span>
                        ))}
                    </div>
                    
                    {/* 파일 목록 제어 버튼들 */}
                    <div className={styles.fileListControls}>
                        <span className={styles.totalInfo}>
                            Total {files.length} files, {totalSizeMB}MB
                        </span>
                        
                        {hasMoreFiles && (
                            <div className={styles.paginationControls}>
                                {!showAllFiles ? (
                                    <>
                                        <button
                                            type="button"
                                            className={styles.paginationBtn}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setShowAllFiles(true);
                                            }}
                                        >
                                            Show All Files
                                            <ChevronDown size={14} />
                                        </button>
                                        
                                        {totalPages > 1 && (
                                            <div className={styles.pageNavigation}>
                                                <button
                                                    type="button"
                                                    className={styles.pageBtn}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setCurrentPage(Math.max(0, currentPage - 1));
                                                    }}
                                                    disabled={currentPage === 0}
                                                >
                                                    <ChevronUp size={12} />
                                                </button>
                                                <span className={styles.pageInfo}>
                                                    {currentPage + 1} / {totalPages}
                                                </span>
                                                <button
                                                    type="button"
                                                    className={styles.pageBtn}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setCurrentPage(Math.min(totalPages - 1, currentPage + 1));
                                                    }}
                                                    disabled={currentPage === totalPages - 1}
                                                >
                                                    <ChevronDown size={12} />
                                                </button>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <button
                                        type="button"
                                        className={styles.paginationBtn}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setShowAllFiles(false);
                                            setCurrentPage(0);
                                        }}
                                    >
                                        Show Less
                                        <ChevronUp size={14} />
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                <span className={styles.placeholder}>
                    {dragActive ? 'Drop files or folders here!' : ' '}
                </span>
            )}
            
            {/* 파일 선택 input */}
            <input
                type="file"
                accept="*"
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple={multiple}
                tabIndex={-1}
            />
            
            {/* 폴더 선택 input */}
            {allowFolders && (
                <input
                    type="file"
                    webkitdirectory=""
                    style={{ display: 'none' }}
                    ref={folderInputRef}
                    onChange={handleFileChange}
                    multiple={multiple}
                    tabIndex={-1}
                />
            )}
            
            {fileError && <div className={styles.errorMsg}>{getFriendlyError(fileError)}</div>}
        </div>
    );
} 