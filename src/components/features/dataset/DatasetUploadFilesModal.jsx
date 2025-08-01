import React, { useState } from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import createModalStyles from '../../ui/modals/CreateModal.module.css';
import FileUploadField from '../../ui/modals/FileUploadField.jsx';

/**
 * 데이터셋 파일 업로드 모달 컴포넌트
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 모달 닫기 핸들러
 * @param {Function} props.onSave - 파일 업로드 핸들러
 */
const DatasetUploadFilesModal = ({ isOpen, onClose, onSave }) => {
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
        <Modal isOpen={isOpen} onClose={onClose} title="Upload Files">
            <form onSubmit={handleSubmit} className={createModalStyles.formGroup} style={{margin:0}}>
                <label className={createModalStyles.label}>
                    <FileUploadField 
                        files={files} 
                        setFiles={setFiles} 
                        fileError={fileError} 
                        setFileError={setFileError} 
                        accept={'.jpg,.jpeg,.png,.gif'} 
                        multiple 
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