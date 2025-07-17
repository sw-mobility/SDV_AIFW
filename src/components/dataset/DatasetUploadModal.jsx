import React, { useState, useRef } from 'react';
import Modal from '../common/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../common/Button.jsx';
import { uploadDataset } from '../../api/datasets.js';
import Loading from '../common/Loading.jsx';
import ErrorMessage from '../common/ErrorMessage.jsx';
import { useDatasetContext } from '../../context/DatasetContext';

const DATASET_TYPES = [
  'Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'
];
const ACCEPTED_FORMATS = '.csv,.xlsx,.xls,.json,.zip';
const MAX_FILE_SIZE_MB = 200; // 200MB 제한

export default function DatasetUploadModal({ isOpen, onClose }) {
  const { reload } = useDatasetContext();
  const [name, setName] = useState('');
  const [type, setType] = useState(DATASET_TYPES[0]);
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    validateAndSetFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const validateAndSetFile = (f) => {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!ACCEPTED_FORMATS.includes(ext) && !ACCEPTED_FORMATS.includes('.' + ext)) {
      setFileError('지원하지 않는 파일 형식입니다. (csv, xlsx, xls, json, zip)');
      setFile(null);
      return;
    }
    if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setFileError(`최대 ${MAX_FILE_SIZE_MB}MB 파일만 업로드할 수 있습니다.`);
      setFile(null);
      return;
    }
    setFile(f);
    setFileError(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleRemoveFile = () => {
    setFile(null);
    setFileError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      // 실제 서비스라면 FormData로 파일 업로드 필요
      await uploadDataset({
        name: name || (file && file.name),
        type,
        size: file ? `${(file.size / (1024 * 1024)).toFixed(1)}MB` : '',
        file
      });
      setSuccess(true);
      reload();
      setTimeout(() => {
        setSuccess(false);
        onClose();
      }, 1000);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setName('');
    setType(DATASET_TYPES[0]);
    setFile(null);
    setFileError(null);
    setError(null);
    setSuccess(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  React.useEffect(() => {
    if (isOpen) resetForm();
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Upload Dataset">
      <form onSubmit={handleSubmit} className={styles.formGroup}>
        <label className={styles.label}>
          Name
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            className={styles.input}
            placeholder="Enter dataset name"
            autoFocus
          />
        </label>
        <label className={styles.label}>
          Type
          <select
            value={type}
            onChange={e => setType(e.target.value)}
            className={styles.input}
          >
            {DATASET_TYPES.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
        <div
          className={styles.uploadDropZone}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
          style={{ cursor: 'pointer' }}
        >
          {file ? (
            <div className={styles.uploadFileInfo}>
              <span>{file.name} ({(file.size / (1024 * 1024)).toFixed(1)}MB)</span>
              <button type="button" className={styles.removeFileButton} onClick={e => { e.stopPropagation(); handleRemoveFile(); }}>Remove</button>
            </div>
          ) : (
            <>
              <span className={styles.uploadDropText}>여기로 파일을 드래그하거나 클릭해서 업로드 (csv, xlsx, xls, json, zip, 최대 {MAX_FILE_SIZE_MB}MB)</span>
              <input
                type="file"
                accept={ACCEPTED_FORMATS}
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
              />
            </>
          )}
        </div>
        {fileError && <div className={styles.fileError}>{fileError}</div>}
        {loading && <Loading />}
        {error && <ErrorMessage message={error} />}
        {success && <div className={styles.successMessage}>Upload complete!</div>}
        <div className={styles.modalActions}>
          <button
            type="button"
            onClick={onClose}
            className={styles.cancelButton}
            disabled={loading}
          >
            Cancel
          </button>
          <Button
            type="submit"
            variant="primary"
            size="medium"
            disabled={loading || !file || fileError}
          >
            Upload
          </Button>
        </div>
      </form>
    </Modal>
  );
} 