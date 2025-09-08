import React, { useState, useEffect } from 'react';
import { PlusCircle, Trash2, Edit3, Folder, Code, Calendar, User } from 'lucide-react';
import { CircularProgress, Box } from '@mui/material';
import Button from '../../ui/atoms/Button.jsx';
import Modal from '../../ui/modals/Modal.jsx';
import styles from './CodebaseManager.module.css';

/**
 * 코드베이스 관리 컴포넌트
 * 코드베이스 목록 조회, 생성, 수정, 삭제 기능 제공
 */
const CodebaseManager = ({ 
  codebases = [], 
  selectedCodebase, 
  onCodebaseSelect, 
  onCodebaseCreate, 
  onCodebaseUpdate, 
  onCodebaseDelete,
  loading = false 
}) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [editingCodebase, setEditingCodebase] = useState(null);
  const [deletingCodebase, setDeletingCodebase] = useState(null);

  // 새 코드베이스 생성 폼 상태
  const [createForm, setCreateForm] = useState({
    name: '',
    algorithm: 'yolo',
    stage: 'training',
    task_type: 'detection',
    description: ''
  });

  // 코드베이스 수정 폼 상태
  const [editForm, setEditForm] = useState({
    name: '',
    algorithm: 'yolo',
    stage: 'training',
    task_type: 'detection',
    description: ''
  });

  // Create new codebase
  const handleCreate = async () => {
    if (!createForm.name.trim()) {
      alert('Please enter a codebase name.');
      return;
    }

    try {
      await onCodebaseCreate(createForm);
      setShowCreateModal(false);
      setCreateForm({
        name: '',
        algorithm: 'yolo',
        stage: 'training',
        task_type: 'detection',
        description: ''
      });
    } catch (error) {
      console.error('Failed to create codebase:', error);
      alert('Failed to create codebase.');
    }
  };

  // Update codebase
  const handleUpdate = async () => {
    if (!editForm.name.trim()) {
      alert('Please enter a codebase name.');
      return;
    }

    try {
      await onCodebaseUpdate(editingCodebase.cid, editForm);
      setShowEditModal(false);
      setEditingCodebase(null);
    } catch (error) {
      console.error('Failed to update codebase:', error);
      alert('Failed to update codebase.');
    }
  };

  // Delete codebase
  const handleDelete = async () => {
    try {
      await onCodebaseDelete(deletingCodebase.cid);
      setShowDeleteModal(false);
      setDeletingCodebase(null);
    } catch (error) {
      console.error('Failed to delete codebase:', error);
      alert('Failed to delete codebase.');
    }
  };

  // 수정 모달 열기
  const openEditModal = (codebase) => {
    setEditingCodebase(codebase);
    setEditForm({
      name: codebase.name || '',
      algorithm: codebase.algorithm || 'yolo',
      stage: codebase.stage || 'training',
      task_type: codebase.task_type || 'detection',
      description: codebase.description || ''
    });
    setShowEditModal(true);
  };

  // 삭제 모달 열기
  const openDeleteModal = (codebase) => {
    setDeletingCodebase(codebase);
    setShowDeleteModal(true);
  };

  // 코드베이스 선택
  const handleCodebaseSelect = (codebase) => {
    onCodebaseSelect(codebase);
  };

  return (
    <div className={styles.codebaseManager}>
      {/* Header */}
      <div className={styles.header}>
        <h3 className={styles.title}>
          <Folder size={20} />
          Codebase Manager
        </h3>
      </div>

      {/* Create Button */}
      <div className={styles.createButtonContainer}>
        <button
          className={styles.createButton}
          onClick={() => setShowCreateModal(true)}
        >
          <PlusCircle size={16} />
          New Codebase
        </button>
      </div>

      {/* Codebase List */}
      <div className={styles.codebaseList}>
        {loading ? (
          <Box 
            display="flex" 
            justifyContent="center" 
            alignItems="center" 
            minHeight="200px"
          >
            <CircularProgress size={32} />
          </Box>
        ) : codebases.length === 0 ? (
          <div className={styles.emptyState}>
            <Code size={40} />
            <p>No codebases available.</p>
            <p>Create a new codebase to get started.</p>
          </div>
        ) : (
          codebases.map((codebase) => (
            <div
              key={codebase.cid}
              className={`${styles.codebaseItem} ${
                selectedCodebase?.cid === codebase.cid ? styles.selected : ''
              }`}
              onClick={() => handleCodebaseSelect(codebase)}
            >
              <div className={styles.codebaseInfo}>
                <div className={styles.codebaseName}>
                  <Code size={14} />
                  {codebase.name || codebase.cid}
                </div>
                <div className={styles.codebaseMeta}>
                  <span className={styles.algorithm}>{codebase.algorithm}</span>
                  <span className={styles.stage}>{codebase.stage}</span>
                  {codebase.task_type && (
                    <span className={styles.taskType}>{codebase.task_type}</span>
                  )}
                </div>
                {codebase.description && (
                  <div className={styles.description}>{codebase.description}</div>
                )}
                <div className={styles.codebaseFooter}>
                  <span className={styles.createdAt}>
                    <Calendar size={12} />
                    {codebase.created_at ? new Date(codebase.created_at).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
              </div>
              <div className={styles.codebaseActions}>
                <button 
                  className={styles.actionButton} 
                  title="Edit" 
                  onClick={(e) => {
                    e.stopPropagation();
                    openEditModal(codebase);
                  }}
                >
                  <Edit3 size={16} />
                </button>
                <button 
                  className={styles.actionButton} 
                  title="Delete" 
                  onClick={(e) => {
                    e.stopPropagation();
                    openDeleteModal(codebase);
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Create New Codebase Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Codebase"
        size="medium"
      >
        <div className={styles.modalContent}>
          <div className={styles.formGroup}>
            <label>Codebase Name *</label>
            <input
              type="text"
              value={createForm.name}
              onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
              placeholder="Enter codebase name"
            />
          </div>
          
          <div className={styles.formGroup}>
            <label>Algorithm</label>
            <select
              value={createForm.algorithm}
              onChange={(e) => setCreateForm({ ...createForm, algorithm: e.target.value })}
            >
              <option value="yolo">YOLO</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Stage</label>
            <select
              value={createForm.stage}
              onChange={(e) => setCreateForm({ ...createForm, stage: e.target.value })}
            >
              <option value="training">Training</option>
              <option value="validation">Validation</option>
              <option value="optimization">Optimization</option>
              <option value="inference">Inference</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Task Type</label>
            <select
              value={createForm.task_type}
              onChange={(e) => setCreateForm({ ...createForm, task_type: e.target.value })}
            >
              <option value="detection">Detection</option>
              <option value="classification">Classification</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <textarea
              value={createForm.description}
              onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
              placeholder="Enter codebase description"
              rows={3}
            />
          </div>

          <div className={styles.modalActions}>
            <Button
              onClick={() => setShowCreateModal(false)}
              variant="secondary"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              variant="primary"
            >
              Create
            </Button>
          </div>
        </div>
      </Modal>

      {/* Edit Codebase Modal */}
      <Modal
        isOpen={showEditModal}
        onClose={() => setShowEditModal(false)}
        title="Edit Codebase"
        size="medium"
      >
        <div className={styles.modalContent}>
          <div className={styles.formGroup}>
            <label>Codebase Name *</label>
            <input
              type="text"
              value={editForm.name}
              onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
              placeholder="Enter codebase name"
            />
          </div>
          
          <div className={styles.formGroup}>
            <label>Algorithm</label>
            <select
              value={editForm.algorithm}
              onChange={(e) => setEditForm({ ...editForm, algorithm: e.target.value })}
            >
              <option value="yolo">YOLO</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Stage</label>
            <select
              value={editForm.stage}
              onChange={(e) => setEditForm({ ...editForm, stage: e.target.value })}
            >
              <option value="training">Training</option>
              <option value="validation">Validation</option>
              <option value="optimization">Optimization</option>
              <option value="inference">Inference</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Task Type</label>
            <select
              value={editForm.task_type}
              onChange={(e) => setEditForm({ ...editForm, task_type: e.target.value })}
            >
              <option value="detection">Detection</option>
              <option value="classification">Classification</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <textarea
              value={editForm.description}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              placeholder="Enter codebase description"
              rows={3}
            />
          </div>

          <div className={styles.modalActions}>
            <Button
              onClick={() => setShowEditModal(false)}
              variant="secondary"
            >
              Cancel
            </Button>
            <Button
              onClick={handleUpdate}
              variant="primary"
            >
              Update
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete Codebase Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Codebase"
        size="small"
      >
        <div style={{ padding: 16, fontSize: 16, color: '#d32f2f', textAlign: 'center' }}>
          <Trash2 size={32} style={{ marginBottom: 8 }} />
          <div>Are you sure you want to delete this codebase?</div>
          <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>
            The codebase <strong>{deletingCodebase?.name || deletingCodebase?.cid}</strong> will be permanently deleted.
          </div>
          <div style={{ fontSize: 14, color: '#888', marginTop: 4 }}>
            This action cannot be undone.
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 16 }}>
          <Button
            onClick={() => setShowDeleteModal(false)}
            variant="secondary"
            size="medium"
          >
            Cancel
          </Button>
          <Button
            onClick={handleDelete}
            variant="danger"
            size="medium"
          >
            Delete
          </Button>
        </div>
      </Modal>
    </div>
  );
};

export default CodebaseManager;
