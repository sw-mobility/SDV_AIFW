import React, { useState, useEffect } from 'react';
import Modal from '../../components/ui/Modal.jsx';
import styles from './CreateModal.module.css';

/**
 * components/ui 의 Modal을 create project modal 로 custom
 *
 * @param isOpen
 * @param onClose
 * @param onSubmit
 * @returns {Element}
 * @constructor
 */
export default function CreateModal({
    isOpen,
    onClose,
    onSubmit,
    initialName = '',
    title = 'Create New Project',
    submitLabel = 'Create Project',
}) {
    const [projectName, setProjectName] = useState(initialName);

    useEffect(() => {
        setProjectName(initialName);
    }, [initialName, isOpen]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (projectName.trim()) {
            onSubmit(projectName.trim());
            setProjectName('');
            onClose();
        }
    };

    const handleClose = () => {
        setProjectName(initialName);
        onClose();
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={handleClose}
            title={title}
        >
            <form onSubmit={handleSubmit}>
                <div className={styles.formGroup}>
                    <label htmlFor="projectName" className={styles.label}>
                        Project Name
                    </label>
                    <input
                        type="text"
                        id="projectName"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        className={styles.input}
                        placeholder="Enter project name"
                        autoFocus
                        required
                    />
                </div>
                <div className={styles.modalActions}>
                    <button
                        type="button"
                        onClick={handleClose}
                        className={styles.cancelButton}
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className={styles.submitButton}
                        disabled={!projectName.trim()}
                    >
                        {submitLabel}
                    </button>
                </div>
            </form>
        </Modal>
    );
} 