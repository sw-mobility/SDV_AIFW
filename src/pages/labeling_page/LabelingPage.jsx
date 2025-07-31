import React from 'react';
import styles from './LabelingPage.module.css';
import DatasetTablePanel from '../../components/features/labeling/DatasetTablePanel.jsx';
import LabelingWorkspace from '../../components/features/labeling/LabelingWorkspace.jsx';
import { useLabeling } from '../../hooks';
const LabelingPage = () => {
    const {
        datasets,
        loading,
        error,
        selectedDataset,
        setSelectedDataset
    } = useLabeling();

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