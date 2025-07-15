import React from 'react';
import styles from './Table.module.css';

/**
 * table component 사용 예시
 *
 * <Table
 *             columns={["Pod", "Status", "CPU", "Memory", "GPU"]}
 *             data={[
 *                 ["Pod 1", "Running", "80%", "60%", "90%"],
 *                 ["Pod 2", "Idle", "10%", "20%", "0%"],
 *             ]}
 *         />
 * @param columns
 * @param data
 * @returns {Element}
 * @constructor
 */
export default function Table({ columns, data }) {
    return (
        <div className={styles['table-container']}>
            <table className={styles.table}>
                <thead className={styles.thead}>
                <tr>
                    {columns.map(col => (
                        <th key={col} className={styles.th}>{col}</th>
                    ))}
                </tr>
                </thead>
                <tbody>
                {data.map((row, idx) => (
                    <tr key={idx} className={styles.tr}>
                        {row.map((cell, i) => (
                            <td key={i} className={styles.td}>{cell}</td>
                        ))}
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
    );
}
