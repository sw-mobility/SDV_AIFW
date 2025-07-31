import React, { useEffect, useState } from 'react';
import styles from './LabelingPage.module.css';
import DatasetTablePanel from '../../components/features/labeling/DatasetTablePanel.jsx';
import LabelingWorkspace from '../../components/features/labeling/LabelingWorkspace.jsx';
import { fetchRawDatasets } from '../../api/datasets.js';
import {uid} from '../../api/uid';
const LabelingPage = () => {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedDataset, setSelectedDataset] = useState(null);

    useEffect(() => {
        setLoading(true);
        fetchRawDatasets({uid})
            .then(res => {
                const camelDatasets = (res.data || []).map(ds => ({
                    ...ds,
                    id: ds.id || ds._id, // id 필드 보장
                    createdAt: ds.created_at ? new Date(ds.created_at).toISOString().slice(0, 10) : undefined
                }));
                setDatasets(camelDatasets);
                setError(null);
            })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className={styles.container}>
            <div className={styles.pageHeader}>
                <h1 className={styles.pageTitle}>Labeling</h1>
                <p className={styles.pageDescription}>
                    Select a dataset and configure your labeling settings
                </p>
            </div>

            {error && (
                <div className={styles.errorMessage}>
                    <span>Error loading datasets: {error}</span>
                </div>
            )}

            <div className={styles.sectionWrap}>
                {/* 좌측: Raw Datasets 패널 */}
                <div className={styles.leftPanel}>
                    <DatasetTablePanel
                        datasets={datasets}
                        selectedId={selectedDataset?.id}
                        onSelect={setSelectedDataset}
                        loading={loading}
                    />
                </div>

                {/* 우측: 라벨링 작업 영역 */}
                <div className={styles.rightPanel}>
                    <LabelingWorkspace dataset={selectedDataset} />
                </div>
            </div>
        </div>
    );
};

export default LabelingPage;