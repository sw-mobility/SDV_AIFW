import React, { useState } from 'react';
import Modal from '../../shared/common/Modal.jsx';
import createModalStyles from '../../shared/common/CreateModal.module.css';
import FileUploadField from '../../shared/common/FileUploadField.jsx';

const UploadFilesModal = ({ isOpen, onClose, onSave }) => {
    const [files, setFiles] = useState([]);
    const [fileError, setFileError] = useState(null);
    const [loading, setLoading] = useState(false);
    const handleSubmit = async e => {
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
                    <FileUploadField files={files} setFiles={setFiles} fileError={fileError} setFileError={setFileError} accept={'.jpg,.jpeg,.png,.gif'} multiple />
                </label>
                {fileError && <div className={createModalStyles.fileError}>{fileError}</div>}
                <div className={createModalStyles.modalActions}>
                    <button type="button" onClick={onClose} className={createModalStyles.cancelButton} disabled={loading}>Cancel</button>
                    <button type="submit" className={createModalStyles.submitButton} disabled={loading || files.length === 0 || fileError}>Upload</button>
                </div>
            </form>
        </Modal>
    );
};

export default UploadFilesModal; 