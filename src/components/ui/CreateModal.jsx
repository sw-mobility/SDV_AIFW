import React, { useState, useEffect } from 'react';
import Modal from './Modal.jsx';
import styles from './CreateModal.module.css';

/**
 * 범용 엔티티 생성/수정 모달
 * @param isOpen
 * @param onClose
 * @param onSubmit
 * @param initialValue
 * @param title
 * @param submitLabel
 * @param label
 * @param placeholder
 * @param inputName
 * @returns {Element}
 */
export default function CreateModal({
    isOpen,
    onClose,
    onSubmit,
    initialValue = '',
    initialDescription = '',
    title = 'Create',
    submitLabel = 'Create',
    label = 'Name',
    placeholder = 'Enter name',
    inputName = 'entityName',
    showDescription = false,
}) {
    const [value, setValue] = useState(initialValue);
    const [description, setDescription] = useState(initialDescription);

    useEffect(() => {
        setValue(initialValue);
        setDescription(initialDescription);
    }, [initialValue, initialDescription, isOpen]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (value.trim()) {
            if (showDescription) {
                onSubmit({ name: value.trim(), description: description.trim() });
            } else {
            onSubmit(value.trim());
            }
            setValue('');
            setDescription('');
            onClose();
        }
    };

    const handleClose = () => {
        setValue(initialValue);
        setDescription(initialDescription);
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
                    <label htmlFor={inputName} className={styles.label}>
                        {label}
                    </label>
                    <input
                        type="text"
                        id={inputName}
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        className={styles.input}
                        placeholder={placeholder}
                        autoFocus
                        required
                    />
                </div>
                {showDescription && (
                    <div className={styles.formGroup}>
                        <label htmlFor="description" className={styles.label}>
                            Description
                        </label>
                        <textarea
                            id="description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            className={styles.textarea}
                            placeholder="Enter description (optional)"
                            rows={3}
                        />
                    </div>
                )}
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
                        disabled={!value.trim()}
                    >
                        {submitLabel}
                    </button>
                </div>
            </form>
        </Modal>
    );
} 