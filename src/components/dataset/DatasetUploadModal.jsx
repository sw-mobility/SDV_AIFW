import React, { useState, useRef } from 'react';
import Modal from '../common/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../common/Button.jsx';
import { uploadDataset, createRawDataset, createLabeledDataset, uploadLabeledFiles } from '../../api/datasets.js';
import Loading from '../common/Loading.jsx';
import ErrorMessage from '../common/ErrorMessage.jsx';
import { useDatasetContext } from '../../context/DatasetContext.jsx';
import { uid } from '../../api/uid.js';
import FileUploadField from '../common/FileUploadField.jsx';

const DATASET_TYPES = [
  'Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'
];
const ACCEPTED_FORMATS = '.csv,.xlsx,.xls,.json,.zip';
const MAX_FILE_SIZE_MB = 200; // 200MB 제한
const ACCEPTED_IMAGE_FORMATS = '.jpg,.jpeg,.png,.gif';

export default function DatasetUploadModal({ isOpen, onClose, datasetType = 'raw', editMode = false, initialData = {}, onSave, onCreated }) {
  const { reload } = useDatasetContext();
  const [name, setName] = useState(initialData.name || '');
  const [type, setType] = useState(initialData.type || DATASET_TYPES[0]);
  const [files, setFiles] = useState([]);
  const [fileError, setFileError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [description, setDescription] = useState(initialData.description || '');
  // Labeled dataset only
  const [taskType, setTaskType] = useState(initialData.task_type || initialData.taskType || 'Classification');
  const [labelFormat, setLabelFormat] = useState(initialData.label_format || initialData.labelFormat || 'COCO');

  React.useEffect(() => {
    if (isOpen) {
      setName(initialData.name || '');
      setType(initialData.type || DATASET_TYPES[0]);
      setDescription(initialData.description || '');
      setTaskType(initialData.task_type || initialData.taskType || 'Classification');
      setLabelFormat(initialData.label_format || initialData.labelFormat || 'COCO');
      setFiles([]);
      setFileError(null);
      setError(null);
      setSuccess(false);
    }
  }, [isOpen]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      if (editMode && onSave) {
        await onSave({ name, type, description, taskType, labelFormat });
      } else if (datasetType === 'labeled') {
        if (files.length > 0) {
          await uploadLabeledFiles({
            files,
            uid: uid,
            did: '', // TODO: set did if available (생성 직후라면 응답에서 받아야 함)
            task_type: taskType,
            label_format: labelFormat
          });
        } else {
          await createLabeledDataset({
            uid: uid,
            name,
            description,
            type,
            task_type: taskType,
            label_format: labelFormat
          });
        }
      } else if (files.length === 0) {
        // Only create dataset meta info (no file)
        await createRawDataset({
          uid: uid,
          name,
          description,
          type
        });
      } else {
        // File upload logic (legacy/mock)
        await uploadDataset({
          name: name || (files[0] && files[0].name),
          type,
          description,
          size: files.length > 0 ? `${(files.reduce((a, f) => a + f.size, 0) / (1024 * 1024)).toFixed(1)}MB` : '',
          file: files[0] // TODO: support multiple file upload in real API
        });
      }
      setSuccess(true);
      reload();
      onCreated && onCreated();
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
    setDescription('');
    setFiles([]);
    setFileError(null);
    setError(null);
    setSuccess(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={datasetType === 'labeled' ? (editMode ? 'Edit Labeled Dataset' : 'Create Labeled Dataset') : (editMode ? 'Edit Dataset' : 'Create Dataset')}>
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
        <label className={styles.label}>
          Description
          <textarea
            value={description}
            onChange={e => setDescription(e.target.value)}
            className={styles.input}
            placeholder="Enter dataset description"
            rows={3}
            style={{ resize: 'vertical' }}
          />
        </label>
        {datasetType === 'labeled' && (
          <>
            <label className={styles.label}>
              Task Type
              <select value={taskType} onChange={e => setTaskType(e.target.value)} className={styles.input}>
                {['Classification', 'Detection', 'Segmentation', 'OCR', 'Other'].map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <label className={styles.label}>
              Label Format
              <select value={labelFormat} onChange={e => setLabelFormat(e.target.value)} className={styles.input}>
                {['COCO', 'VOC', 'YOLO', 'Custom'].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </label>
          </>
        )}
        {loading && <Loading />}
        {error && <ErrorMessage message={error} />}
        {success && <div className={styles.successMessage}>{editMode ? 'Saved!' : 'Upload complete!'}</div>}
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
            disabled={loading || (datasetType === 'raw' && !editMode && files.length === 0 && !name) || fileError}
          >
            {editMode ? 'Save' : 'Create'}
          </Button>
        </div>
      </form>
    </Modal>
  );
} 