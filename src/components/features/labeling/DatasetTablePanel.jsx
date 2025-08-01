import React from 'react';
import { Database, Loader2 } from 'lucide-react';
import DatasetTable from '../dataset/RawDatasetTable.jsx';
import styles from './DatasetTablePanel.module.css';

/**
 * 라벨링 페이지에서 데이터셋 테이블을 표시하는 패널
 * features/dataset 의 rawdatasettable 컴포넌트 정보 포함
 * 주요 기능:
 * 데이터셋 목록 표시
 * 라벨링 대상 데이터셋 선택
 *
 * @param datasets
 * @param selectedId
 * @param onSelect
 * @param loading
 * @returns {Element}
 * @constructor
 */
export default function DatasetTablePanel({ datasets, selectedId, onSelect, loading }) {
    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <div className={styles.headerLeft}>
                    <Database size={20} color="#4f8cff" />
                    <span className={styles.title}>Raw Datasets</span>
                </div>
                <div className={styles.headerRight}>
                    {loading && (
                        <div className={styles.loadingIndicator}>
                            <Loader2 size={16} className={styles.spinner} />
                        </div>
                    )}
                    <span className={styles.count}>
            {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}
          </span>
                </div>
            </div>

            <div className={styles.tableContainer}>
                <DatasetTable
                    columns={["Name", "Created At"]}
                    data={datasets}
                    onRowClick={onSelect}
                    selectedId={selectedId}
                    searchPlaceholder="Search dataset name..."
                    loading={loading}
                />
            </div>
        </div>
    );
}