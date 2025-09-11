import React, { useState } from 'react';
import Modal from './Modal.jsx';
import Button from '../atoms/Button.jsx';
import styles from '../../features/dataset/Dataset.module.css';

const SnapshotModal = ({ isOpen, onClose, onSave, algorithm, defaultName = '' }) => {
  const [formData, setFormData] = useState({
    name: defaultName || `${algorithm} Snapshot`,
    description: '',
    stage: 'training',
    task_type: 'detection'
  });

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  const handleClose = () => {
    setFormData({
      name: defaultName || `${algorithm} Snapshot`,
      description: '',
      stage: 'training',
      task_type: 'detection'
    });
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Save Snapshot">
      <form onSubmit={handleSubmit} className={styles.formGroup}>
        <label className={styles.label}>
          Snapshot Name *
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
            className={styles.input}
            placeholder="Enter snapshot name"
            required
          />
        </label>

        <label className={styles.label}>
          Description
          <textarea
            value={formData.description}
            onChange={(e) => handleInputChange('description', e.target.value)}
            className={styles.input}
            placeholder="Enter description (optional)"
            rows={3}
            style={{resize: 'vertical'}}
          />
        </label>

        <label className={styles.label}>
          Stage
          <select
            value={formData.stage}
            onChange={(e) => handleInputChange('stage', e.target.value)}
            className={styles.input}
          >
            <option value="training">Training</option>
            <option value="validation">Validation</option>
            <option value="testing">Testing</option>
            <option value="deployment">Deployment</option>
          </select>
        </label>

        <label className={styles.label}>
          Task Type
          <select
            value={formData.task_type}
            onChange={(e) => handleInputChange('task_type', e.target.value)}
            className={styles.input}
          >
            <option value="detection">Detection</option>
            <option value="classification">Classification</option>
            <option value="segmentation">Segmentation</option>
            <option value="pose">Pose Estimation</option>
          </select>
        </label>

        <div className={styles.modalActions}>
          <button
            type="button"
            onClick={handleClose}
            className={styles.cancelButton}
            disabled={false}
          >
            Cancel
          </button>
          <Button
            type="submit"
            variant="primary"
            size="medium"
            disabled={!formData.name.trim()}
          >
            Save Snapshot
          </Button>
        </div>
      </form>
    </Modal>
  );
};

export default SnapshotModal;
