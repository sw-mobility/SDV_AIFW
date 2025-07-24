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
export default function Table({ columns, data, onRowClick, selectedId, rowKey = 'id', selectedRowClassName }) {
    // 생성일자 컬럼 인덱스 찾기
    const createdAtIdx = columns.findIndex(col => col.toLowerCase().includes('created'));
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
                {data.length === 0 ? (
                    <tr>
                        <td colSpan={columns.length} className={styles.td} style={{ color: '#aaa', textAlign: 'center' }}>
                            Not found.
                        </td>
                    </tr>
                ) : (
                    data.map((row, idx) => (
                        <tr
                            key={rowKey && row[rowKey] ? row[rowKey] : idx}
                            className={
                              styles.tr +
                              (selectedId && rowKey && row[rowKey] === selectedId
                                ? ' ' + (selectedRowClassName || styles.selectedRow)
                                : '')
                            }
                            onClick={() => onRowClick && onRowClick(row, idx)}
                        >
                            {row.cell !== undefined
                              ? <td className={styles.td}>{row.cell}</td>
                              : row.map((cell, i) => (
                                  <td
                                    key={i}
                                    className={i === createdAtIdx ? styles.tdCreatedAt : styles.td}
                                  >
                                    {cell}
                                  </td>
                                ))}
                        </tr>
                    ))
                )}
                </tbody>
            </table>
        </div>
    );
}
