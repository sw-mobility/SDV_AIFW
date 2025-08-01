import React from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import Button from '../../ui/atoms/Button.jsx';
import styles from './Dataset.module.css';
import { updateRawDataset, updateLabeledDataset } from '../../../api/datasets.js';

/**
 * 데이터셋 편집 전용 모달
 * 주요 기능:
 * Raw와 Labeled 데이터셋 모두 편집 가능
 * 이름, 타입, 설명 수정
 * Labeled 데이터셋일 때 Task Type, Label Format 추가 필드
 * API를 통한 데이터셋 업데이트
 *
 * @param open
 * @param onClose
 * @param dataset
 * @param datasetType - 'raw' 또는 'labeled'
 * @param onUpdated
 * @returns {React.JSX.Element|null}
 * @constructor
 */
const DatasetEditModal = ({open, onClose, dataset, datasetType = 'raw', onUpdated}) => {
    const [name, setName] = React.useState(dataset?.name || '');
    const [type, setType] = React.useState(dataset?.type || 'Image');
    const [description, setDescription] = React.useState(dataset?.description || '');
    const [taskType, setTaskType] = React.useState(dataset?.task_type || 'Classification');
    const [labelFormat, setLabelFormat] = React.useState(dataset?.label_format || 'COCO');
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState(null);

    React.useEffect(() => {
        if (open) {
            setName(dataset?.name || '');
            setType(dataset?.type || 'Image');
            setDescription(dataset?.description || '');
            setTaskType(dataset?.task_type || 'Classification');
            setLabelFormat(dataset?.label_format || 'COCO');
            setError(null);
        }
    }, [open, dataset]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            if (datasetType === 'labeled') {
                await updateLabeledDataset({
                    id: dataset._id,
                    uid: dataset.uid,
                    name,
                    description,
                    type,
                    task_type: taskType,
                    label_format: labelFormat
                });
            } else {
                await updateRawDataset({
                    id: dataset._id,
                    uid: dataset.uid,
                    name,
                    description,
                    type
                });
            }
            onUpdated && onUpdated();
            onClose();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (!open) return null;

    return (
        <Modal 
            isOpen={open} 
            onClose={onClose} 
            title={`Edit ${datasetType === 'labeled' ? 'Labeled' : ''} Dataset`}
        >
            <form onSubmit={handleSubmit} className={styles.formGroup}>
                <label className={styles.label}>
                    Name
                    <input 
                        type="text" 
                        value={name} 
                        onChange={e => setName(e.target.value)}
                        className={styles.input}
                        placeholder="Enter dataset name"
                    />
                </label>
                <label className={styles.label}>
                    Type
                    <select 
                        value={type} 
                        onChange={e => setType(e.target.value)} 
                        className={styles.input}
                    >
                        {['Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'].map(t => (
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
                        rows={3} 
                        style={{resize: 'vertical'}}
                        placeholder="Enter dataset description"
                    />
                </label>
                
                {/* Labeled 데이터셋일 때만 추가 필드 표시 */}
                {datasetType === 'labeled' && (
                    <>
                        <label className={styles.label}>
                            Task Type
                            <select 
                                value={taskType} 
                                onChange={e => setTaskType(e.target.value)} 
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
                                value={labelFormat} 
                                onChange={e => setLabelFormat(e.target.value)} 
                                className={styles.input}
                            >
                                {['COCO', 'VOC', 'YOLO', 'Custom'].map(f => (
                                    <option key={f} value={f}>{f}</option>
                                ))}
                            </select>
                        </label>
                    </>
                )}
                
                {error && <div className={styles.fileError}>{error}</div>}
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
                        disabled={loading || !name}
                    >
                        Save
                    </Button>
                </div>
            </form>
        </Modal>
    );
};

export default DatasetEditModal; 