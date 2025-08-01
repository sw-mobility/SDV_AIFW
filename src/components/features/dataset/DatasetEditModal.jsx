import React from 'react';
import Button from '@mui/material/Button';
import styles from './Dataset.module.css';
import { updateRawDataset } from '../../../api/datasets.js';
import * as editTarget from "../../../api/uid.js";

const DatasetEditModal = ({open, onClose, dataset, onUpdated}) => {
    const [name, setName] = React.useState(dataset?.name || '');
    const [type, setType] = React.useState(dataset?.type || 'Image');
    const [description, setDescription] = React.useState(dataset?.description || '');
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState(null);
    React.useEffect(() => {
        if (open) {
            setName(dataset?.name || '');
            setType(dataset?.type || 'Image');
            setDescription(dataset?.description || '');
            setError(null);
        }
    }, [open, dataset]);
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            await updateRawDataset({
                id: dataset._id,
                uid: dataset.uid,
                name,
                description,
                type
            });
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
        <div className={styles['modals-backdrop']}>
            <div className={styles['modal']}>
                <form onSubmit={handleSubmit} className={styles.formGroup}>
                    <div style={{fontWeight: 600, fontSize: 18, marginBottom: 12}}>Edit Dataset</div>
                    <label className={styles.label}>
                        Name
                        <input type="text" value={name} onChange={e => setName(e.target.value)}
                               className={styles.input}/>
                    </label>
                    <label className={styles.label}>
                        Type
                        <select value={type} onChange={e => setType(e.target.value)} className={styles.input}>
                            {['Image', 'Text', 'Audio', 'Video', 'Tabular', 'TimeSeries', 'Graph'].map(t => (
                                <option key={t} value={t}>{t}</option>
                            ))}
                        </select>
                    </label>
                    <label className={styles.label}>
                        Description
                        <textarea value={description} onChange={e => setDescription(e.target.value)}
                                  className={styles.input} rows={3} style={{resize: 'vertical'}}/>
                    </label>
                    {error && <div className={styles.fileError}>{error}</div>}
                    <div className={styles.modalActions}>
                        <button type="button" onClick={onClose} className={styles.cancelButton}
                                disabled={loading}>Cancel
                        </button>
                        <Button type="submit" variant="primary" size="medium" disabled={loading || !name}>Save</Button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default DatasetEditModal; 