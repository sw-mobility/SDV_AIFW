import React, { useRef, useState } from 'react';
import styles from './FileUploadField.module.css';
import { Upload } from 'lucide-react';

/**
 * 파일 업로드를 위한 field
 * @param files
 * @param setFiles
 * @param fileError
 * @param setFileError
 * @param accept
 * @param multiple
 * @param maxFiles
 * @param maxSizeMB
 * @returns {Element}
 * @constructor
 */
export default function FileUploadField({ files, setFiles, fileError, setFileError, accept = '*', multiple = true, maxFiles = 20, maxSizeMB = 100 }) {
    const fileInputRef = useRef();
    const [dragActive, setDragActive] = useState(false);
    const validateFiles = (selected) => {
        if (accept && accept !== '*') {
            const allowed = accept.split(',').map(s => s.replace('.', '').toLowerCase());
            const invalid = selected.find(f => !allowed.includes(f.name.split('.').pop().toLowerCase()));
            if (invalid) {
                setFileError(`허용된 파일 형식만 업로드할 수 있습니다: ${accept}`);
                return false;
            }
        }
        if (files.length + selected.length > maxFiles) {
            setFileError(`최대 ${maxFiles}개 파일만 업로드할 수 있습니다.`);
            return false;
        }
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

    return (
        <div
            className={[
                styles.uploadField,
                dragActive ? styles.dragActive : '',
                fileError ? styles.error : '',
            ].join(' ')}
            tabIndex={0}
            aria-label="파일 업로드 영역"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDragEnd={handleDragLeave}
        >
            <div className={styles.headerRow}>
                <span className={styles.iconWrap}><Upload size={22} /></span>
                <span className={styles.title}>Drag or click multiple files to upload</span>
                <span className={styles.info}>
                    (max {maxFiles}, {maxSizeMB}MB)
                </span>
            </div>
            {files && files.length > 0 ? (
                <div className={styles.fileList}>
                    {files.map((f) => (
                        <span key={f.name + f.size} className={styles.fileTag}>
                            <span className={styles.fileName} title={f.name}>{f.name}</span>
                            <span className={styles.fileSize}>({(f.size / (1024 * 1024)).toFixed(1)}MB)</span>
                            <button type="button" aria-label="파일 삭제" className={styles.removeBtn} onClick={e => { e.stopPropagation(); handleRemoveFile(f); }}>×</button>
                        </span>
                    ))}
                    <span className={styles.totalInfo}>Total {files.length}, {totalSizeMB}MB</span>
                </div>
            ) : (
                <span className={styles.placeholder}>
                    {dragActive ? 'Drop files here!' : ' '}
                </span>
            )}
            <input
                type="file"
                accept={accept === '*' ? undefined : accept}
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple={multiple}
                tabIndex={-1}
            />
            {fileError && <div className={styles.errorMsg}>{getFriendlyError(fileError)}</div>}
        </div>
    );
} 