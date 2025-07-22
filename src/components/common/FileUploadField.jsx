import React, { useRef } from 'react';
import styles from './CreateModal.module.css';

// 공용 파일 업로드 필드
export default function FileUploadField({ files, setFiles, fileError, setFileError, accept = '.jpg,.jpeg,.png,.gif', multiple = true }) {
    const fileInputRef = useRef();
    const handleFileChange = (e) => {
        const selected = Array.from(e.target.files);
        // 확장자 체크
        if (accept) {
            const allowed = accept.split(',').map(s => s.replace('.', '').toLowerCase());
            const invalid = selected.find(f => !allowed.includes(f.name.split('.').pop().toLowerCase()));
            if (invalid) {
                setFileError(`허용된 파일 형식만 업로드할 수 있습니다: ${accept}`);
                return;
            }
        }
        setFiles(selected);
        setFileError && setFileError(null);
    };
    return (
        <div
            className={styles.input}
            style={{ minHeight: 80, background: '#f8fafc', border: '2px dashed #cbd5e1', cursor: 'pointer', padding: 16, marginBottom: 8 }}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
        >
            {files && files.length > 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, color: '#222', flex: 1 }}>{files.map(f => f.name).join(', ')} ({(files.reduce((a, f) => a + f.size, 0) / (1024 * 1024)).toFixed(1)}MB)</span>
                    <button type="button" style={{ marginLeft: 8, color: '#e11d48', background: 'none', border: 'none', cursor: 'pointer', fontSize: 16 }} onClick={e => { e.stopPropagation(); setFiles([]); setFileError && setFileError(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}>×</button>
                </div>
            ) : (
                <span style={{ color: '#888', fontSize: 13 }}>
                    여기에 파일을 드래그하거나 클릭해서 업로드 ({accept}, 여러 개 가능)
                </span>
            )}
            <input
                type="file"
                accept={accept}
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple={multiple}
            />
        </div>
    );
} 