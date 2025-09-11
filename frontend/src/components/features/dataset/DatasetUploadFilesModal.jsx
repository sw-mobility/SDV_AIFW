import React, { useState } from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import createModalStyles from '../../ui/modals/CreateModal.module.css';
import FileUploadField from './FileUploadField.jsx';

/**
 * 데이터셋에 파일을 업로드하는 모달
 * 주요 기능:
 * 다중 파일 업로드 (무제한)
 * 폴더 업로드 지원
 * 모든 파일 형식 허용
 * 업로드 진행 상태 표시
 * 배치 업로드 진행률 표시
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 모달 닫기 핸들러
 * @param {Function} props.onSave - 파일 업로드 핸들러
 * @param {Object} props.uploadProgress - 업로드 진행률 정보 (배치 업로드용)
 */
const DatasetUploadFilesModal = ({ isOpen, onClose, onSave, uploadProgress = null }) => {
    const [files, setFiles] = useState([]);
    const [fileError, setFileError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        await onSave(files);
        setLoading(false);
        setFiles([]);
        setFileError(null);
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Upload Files & Folders">
            <form onSubmit={handleSubmit} className={createModalStyles.formGroup} style={{margin:0}}>
                <label className={createModalStyles.label}>
                    <FileUploadField 
                        files={files} 
                        setFiles={setFiles} 
                        fileError={fileError} 
                        setFileError={setFileError} 
                        accept="*" 
                        multiple={true}
                        allowFolders={true} // 폴더 선택 허용
                        uploadProgress={uploadProgress} // 배치 업로드 진행률
                    />
                </label>
                {fileError && <div className={createModalStyles.fileError}>{fileError}</div>}
                <div className={createModalStyles.modalActions}>
                    <button 
                        type="button" 
                        onClick={onClose} 
                        className={createModalStyles.cancelButton} 
                        disabled={loading}
                    >
                        Cancel
                    </button>
                    <button 
                        type="submit" 
                        className={createModalStyles.submitButton} 
                        disabled={loading || files.length === 0 || fileError}
                    >
                        Upload
                    </button>
                </div>
            </form>
        </Modal>
    );
};

export default DatasetUploadFilesModal; 