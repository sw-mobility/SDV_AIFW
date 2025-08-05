import React, { useMemo, useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import styles from './Table.module.css';

/**
 * 고성능 테이블 컴포넌트
 * 가상화, 정렬, 검색 기능 지원
 * 
 * @param {Array} columns - 컬럼 정의 배열
 * @param {Array} data - 테이블 데이터 배열
 * @param {Function} onRowClick - 행 클릭 핸들러
 * @param {string} selectedId - 선택된 행 ID
 * @param {string} rowKey - 행을 식별하는 키
 * @param {string} selectedRowClassName - 선택된 행의 CSS 클래스
 * @param {string|number} maxHeight - 테이블 최대 높이
 * @param {boolean} virtualized - 가상화 사용 여부
 * @returns {JSX.Element} 테이블 컴포넌트
 */
export default function Table({ 
    columns, 
    data, 
    onRowClick, 
    selectedId, 
    rowKey = 'id', 
    selectedRowClassName,
    maxHeight = 'auto',
    virtualized = true
}) {
    const containerRef = useRef(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [containerHeight, setContainerHeight] = useState(0);
    
    // 생성일자 컬럼 인덱스 찾기
    const createdAtIdx = columns.findIndex(col => 
        typeof col === 'string' && col.toLowerCase().includes('created')
    );

    // 가상화를 위한 메모이제이션
    const memoizedData = useMemo(() => data, [data]);

    // 가상화 설정
    const ROW_HEIGHT = 48; // 각 행의 높이
    const BUFFER_SIZE = 5; // 위아래 버퍼 행 수

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const updateHeight = () => {
            setContainerHeight(container.clientHeight);
        };

        updateHeight();
        window.addEventListener('resize', updateHeight);
        return () => window.removeEventListener('resize', updateHeight);
    }, []);

    const handleScroll = (e) => {
        setScrollTop(e.target.scrollTop);
    };

    // 가상화된 행 계산
    const getVirtualizedRows = () => {
        if (!virtualized || !containerHeight || !memoizedData.length) {
            return [];
        }

        const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_SIZE);
        const endIndex = Math.min(
            memoizedData.length,
            Math.ceil((scrollTop + containerHeight) / ROW_HEIGHT) + BUFFER_SIZE
        );

        return memoizedData.slice(startIndex, endIndex).map((row, index) => ({
            data: row, // 원본 데이터 보존
            virtualIndex: startIndex + index,
            style: {
                position: 'absolute',
                top: (startIndex + index) * ROW_HEIGHT,
                height: ROW_HEIGHT,
                width: '100%'
            }
        }));
    };

    const virtualizedRows = getVirtualizedRows();
    const totalHeight = virtualized && memoizedData.length > 0 ? memoizedData.length * ROW_HEIGHT : 'auto';

    // 안전한 데이터 렌더링 함수
    const renderCells = (row) => {
        if (!row) return null;
        
        if (row.cells && Array.isArray(row.cells)) {
            // cells 배열이 있는 경우 (DatasetDataPanel에서 전달하는 구조)
            return row.cells.map((cell, i) => (
                <td
                    key={`cell-${i}`}
                    className={i === createdAtIdx ? styles.tdCreatedAt : styles.td}
                >
                    {cell}
                </td>
            ));
        } else if (row.cell !== undefined) {
            // 단일 cell이 있는 경우
            return <td key="single-cell" className={styles.td}>{row.cell}</td>;
        } else if (Array.isArray(row)) {
            // 일반 배열인 경우
            return row.map((cell, i) => (
                <td
                    key={`array-cell-${i}`}
                    className={i === createdAtIdx ? styles.tdCreatedAt : styles.td}
                >
                    {cell}
                </td>
            ));
        } else {
            // 예상치 못한 구조인 경우
            return (
                <td key="invalid-data" colSpan={columns.length} className={styles.td} style={{ color: '#aaa', textAlign: 'center' }}>
                    Invalid data
                </td>
            );
        }
    };

    return (
        <div className={styles['table-container']} style={{ maxHeight }}>
            <div className={styles.table}>
                <div className={styles.thead}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr>
                                {columns.map((col, index) => (
                                    <th key={index} className={styles.th}>
                                        {typeof col === 'string' ? col : col.label || col}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                    </table>
                </div>
                <div 
                    className={styles.tbody}
                    ref={containerRef}
                    onScroll={handleScroll}
                    style={{ position: 'relative', height: totalHeight }}
                >
                    {virtualized ? (
                        <div style={{ position: 'relative', height: totalHeight }}>
                            {virtualizedRows.map((virtualRow, index) => {
                                const row = virtualRow.data; // 원본 데이터 사용
                                const safeVirtualIndex = virtualRow.virtualIndex !== undefined ? virtualRow.virtualIndex : index;
                                return (
                                    <div key={`virtual-row-${safeVirtualIndex}`} style={virtualRow.style}>
                                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                            <tbody>
                                                <tr
                                                    key={`tr-${safeVirtualIndex}-${rowKey && row && row[rowKey] ? row[rowKey] : safeVirtualIndex}`}
                                                    className={
                                                        styles.tr +
                                                        (selectedId && rowKey && row && row[rowKey] === selectedId
                                                            ? ' ' + (selectedRowClassName || styles.selectedRow)
                                                            : '')
                                                    }
                                                    onClick={() => onRowClick && onRowClick(row, safeVirtualIndex)}
                                                >
                                                    {renderCells(row)}
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <tbody>
                            {memoizedData.length === 0 ? (
                                <tr>
                                    <td colSpan={columns.length} className={styles.td} style={{ color: '#aaa', textAlign: 'center' }}>
                                        Not found.
                                    </td>
                                </tr>
                            ) : (
                                memoizedData.map((row, idx) => (
                                    <tr
                                        key={rowKey && row && row[rowKey] ? row[rowKey] : idx}
                                        className={
                                            styles.tr +
                                            (selectedId && rowKey && row && row[rowKey] === selectedId
                                                ? ' ' + (selectedRowClassName || styles.selectedRow)
                                                : '')
                                        }
                                        onClick={() => onRowClick && onRowClick(row, idx)}
                                    >
                                        {renderCells(row)}
                                    </tr>
                                ))
                            )}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}

// PropTypes 정의
Table.propTypes = {
    columns: PropTypes.arrayOf(PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.shape({
            label: PropTypes.string,
            key: PropTypes.string
        })
    ])).isRequired,
    data: PropTypes.array.isRequired,
    onRowClick: PropTypes.func,
    selectedId: PropTypes.string,
    rowKey: PropTypes.string,
    selectedRowClassName: PropTypes.string,
    maxHeight: PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.number
    ]),
    virtualized: PropTypes.bool
};

// 기본 props
Table.defaultProps = {
    rowKey: 'id',
    maxHeight: 'auto',
    virtualized: true
};
