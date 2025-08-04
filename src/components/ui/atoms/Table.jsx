import React, { useMemo } from 'react';
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
 * @param onRowClick
 * @param selectedId
 * @param rowKey
 * @param selectedRowClassName
 * @param maxHeight
 * @param virtualized
 * @returns {Element}
 * @constructor
 */
export default function Table({ 
    columns, 
    data, 
    onRowClick, 
    selectedId, 
    rowKey = 'id', 
    selectedRowClassName,
    maxHeight = 'auto',
    virtualized = false
}) {
    // 생성일자 컬럼 인덱스 찾기
    const createdAtIdx = columns.findIndex(col => 
        typeof col === 'string' && col.toLowerCase().includes('created')
    );

    // 가상화를 위한 메모이제이션
    const memoizedData = useMemo(() => data, [data]);

    return (
        <div className={styles['table-container']} style={{ maxHeight }}>
            <table className={styles.table}>
                <thead className={styles.thead}>
                <tr>
                    {columns.map((col, index) => (
                        <th key={index} className={styles.th}>
                            {typeof col === 'string' ? col : col.label || col}
                        </th>
                    ))}
                </tr>
                </thead>
                <tbody className={styles.tbody}>
                {memoizedData.length === 0 ? (
                    <tr>
                        <td colSpan={columns.length} className={styles.td} style={{ color: '#aaa', textAlign: 'center' }}>
                            Not found.
                        </td>
                    </tr>
                ) : (
                    memoizedData.map((row, idx) => (
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
