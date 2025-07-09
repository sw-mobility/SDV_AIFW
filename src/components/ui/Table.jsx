import React from 'react';
import './Table.module.css';

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
        <div className="table-container">
            <table>
                <thead>
                <tr>
                    {columns.map(col => (
                        <th key={col}>{col}</th>
                    ))}
                </tr>
                </thead>
                <tbody>
                {data.map((row, idx) => (
                    <tr key={idx}>
                        {row.map((cell, i) => (
                            <td key={i}>{cell}</td>
                        ))}
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
    );
}
