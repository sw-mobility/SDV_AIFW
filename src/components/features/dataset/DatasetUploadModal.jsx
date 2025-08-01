import React from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../../ui/atoms/Button.jsx';
import Loading from '../../ui/atoms/Loading.jsx';
import ErrorMessage from '../../ui/atoms/ErrorMessage.jsx';
import { useDatasetUpload } from '../../../hooks/dataset/useDatasetUpload.js';

export default function DatasetUploadModal({ isOpen, onClose, datasetType = 'raw', editMode = false, initialData = {}, onSave, onCreated }) {
  const {
    formData,
    loading,
    error,
    success,
    updateFormData,
    handleSubmit,
    DATASET_TYPES
  } = useDatasetUpload(initialData, editMode, datasetType, onCreated);

  const onSubmit = async (e) => {
    e.preventDefault();
    
    if (editMode && onSave) {
      await onSave(formData);
      onClose();
    } else {
      await handleSubmit(e);
    }
  };

  // 성공 시 모달 닫기
  React.useEffect(() => {
    if (success) {
      setTimeout(() => {
        onClose();
      }, 1000);
    }
  }, [success, onClose]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={datasetType === 'labeled' ? (editMode ? 'Edit Labeled Dataset' : 'Create Labeled Dataset') : (editMode ? 'Edit Dataset' : 'Create Dataset')}>
      <form onSubmit={onSubmit} className={styles.formGroup}>
        <label className={styles.label}>
          Name
          <input
            type="text"
            value={formData.name}
            onChange={e => updateFormData('name', e.target.value)}
            className={styles.input}
            placeholder="Enter dataset name"
            autoFocus
          />
        </label>
        <label className={styles.label}>
          Type
          <select
            value={formData.type}
            onChange={e => updateFormData('type', e.target.value)}
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
            value={formData.description}
            onChange={e => updateFormData('description', e.target.value)}
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
              <select 
                value={formData.taskType} 
                onChange={e => updateFormData('taskType', e.target.value)} 
                className={styles.input}
              >
                {['Classification', 'Detection', 'Segmentation', 'OCR', 'Other'].map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <label className={styles.label}>
              Label Format
              <select 
                value={formData.labelFormat} 
                onChange={e => updateFormData('labelFormat', e.target.value)} 
                className={styles.input}
              >
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
            disabled={loading || !formData.name}
          >
            {editMode ? 'Save' : 'Create'}
          </Button>
        </div>
      </form>
    </Modal>
  );
} 