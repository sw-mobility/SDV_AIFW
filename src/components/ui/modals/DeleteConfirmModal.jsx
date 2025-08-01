import React from 'react';
import Modal from './Modal.jsx';
import Button from '../atoms/Button.jsx';
import { Trash2 } from 'lucide-react';
import styles from '../../features/dataset/Dataset.module.css';

/**
 * 삭제 전 사용자에게 삭제여부를 한번 더 확인하는 modal
 *
 * @param isOpen
 * @param onClose
 * @param onConfirm
 * @param title
 * @param message
 * @param confirmText
 * @param itemName
 * @returns {Element}
 * @constructor
 */
const DeleteConfirmModal = ({ 
    isOpen, 
    onClose, 
    onConfirm, 
    title = "Delete Item",
    message = "Are you sure you want to delete this item?",
    confirmText = "Delete",
    itemName = ""
}) => {
    return (
        <Modal isOpen={isOpen} onClose={onClose} title={title} className={styles.confirmModal}>
            <div style={{ padding: 16, fontSize: 16, color: '#d32f2f', textAlign: 'center' }}>
                <Trash2 size={32} style={{ marginBottom: 8 }} />
                <div>{message}</div>
                {itemName && (
                    <div style={{ fontSize: 14, color: '#666', marginTop: 8, fontWeight: 600 }}>
                        "{itemName}"
                    </div>
                )}
                <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>
                    This action cannot be undone.
                </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 16 }}>
                <Button variant="secondary" onClick={onClose} size="medium">
                    Cancel
                </Button>
                <Button variant="danger" onClick={onConfirm} size="medium">
                    {confirmText}
                </Button>
            </div>
        </Modal>
    );
};

export default DeleteConfirmModal; 