import React, { useState, useMemo } from 'react';
import Modal from '../../ui/modals/Modal.jsx';
import styles from './Dataset.module.css';
import Button from '../../ui/atoms/Button.jsx';
import Loading from '../../ui/atoms/Loading.jsx';
import ErrorMessage from '../../ui/atoms/ErrorMessage.jsx';
import * as TableModule from '../../ui/atoms/Table.jsx';
import FileUploadField from './FileUploadField.jsx';
import { useDatasetData } from '../../../hooks/dataset/useDatasetData.js';
import { Trash2, Download, Database, Tag, Search } from 'lucide-react';
import { getFileNameOnly } from '../../../utils/fileUtils.js';

const Table = TableModule.default;

/**
 * 데이터셋 클릭 시 나타나는 상세 정보 모달
 * 주요 기능:
 * 데이터셋 내부 데이터 목록 표시
 * 데이터 업로드 기능
 * 선택된 데이터 다운로드/삭제
 * 데이터셋 상세 정보 표시
 * 검색 및 Select All 기능
 *
 * @param open
 * @param onClose
 * @param dataset
 * @returns {Element}
 * @constructor
 */

const DatasetDataPanel = ({ open, onClose, dataset }) => {
    const [search, setSearch] = useState('');
    
    const {
        data,
        loading,
        error,
        selected,
        setSelected, // setSelected 함수 추가
        uploading,
        uploadError,
        uploadFiles,
        showDeleteConfirm,
        downloading,
        handleSelect,
        handleSelectAll: originalHandleSelectAll,
        handleDelete,
        handleUpload,
        handleDownloadSelected,
        updateUploadFiles,
        toggleDeleteConfirm,
        isLabeled,
        uploadProgress // 배치 업로드 진행률 추가
    } = useDatasetData(dataset, open);

    const titleIcon = isLabeled ? <Tag size={20} /> : <Database size={20} />;
    const [debouncedSearch, setDebouncedSearch] = useState(search);
    
    React.useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedSearch(search);
        }, 150); // 150ms 디바운싱
        
        return () => clearTimeout(timer);
    }, [search]);

    const filteredData = useMemo(() => {
        if (!data?.data_list || !Array.isArray(data.data_list)) {
            return [];
        }
        
        if (!debouncedSearch) return data.data_list;
        
        return data.data_list.filter(row => 
            row.name?.toLowerCase().includes(debouncedSearch.toLowerCase()) ||
            row.type?.toLowerCase().includes(debouncedSearch.toLowerCase()) ||
            row.file_format?.toLowerCase().includes(debouncedSearch.toLowerCase()) ||
            (row.created_at && new Date(row.created_at).toLocaleString().toLowerCase().includes(debouncedSearch.toLowerCase()))
        );
    }, [data?.data_list, debouncedSearch]);

    const handleSelectAll = React.useCallback(() => {
        if (!filteredData || !Array.isArray(filteredData)) return;
        
        const filteredIds = filteredData.map(d => d._id).filter(Boolean);
        const allFilteredSelected = filteredIds.every(id => selected.includes(id));
        
        if (allFilteredSelected) {
            // 모든 검색된 항목이 선택되어 있으면 선택 해제
            setSelected(prev => prev.filter(id => !filteredIds.includes(id)));
        } else {
            // 검색된 항목 중 선택되지 않은 것들을 선택
            setSelected(prev => [...new Set([...prev, ...filteredIds])]);
        }
    }, [filteredData, selected]);
    const handleCheckboxSelect = React.useCallback((row) => {
        setSelected(prev => {
            const newSelected = prev.includes(row._id) 
                ? prev.filter(id => id !== row._id)
                : [...prev, row._id];
            return newSelected;
        });
    }, []);

    // 체크박스 상태를 Set으로 관리 성능 최적화
    const selectedSet = React.useMemo(() => new Set(selected), [selected]);

    // 체크박스 컴포넌트를 별도로 생성 성능 최적화
    const CheckboxCell = React.memo(({ rowId, isSelected, onSelect }) => (
        <input 
            type="checkbox" 
            checked={isSelected} 
            onChange={e => { 
                e.stopPropagation(); 
                e.preventDefault();
                onSelect(); 
            }} 
            onClick={e => e.stopPropagation()} 
            style={{ cursor: 'pointer' }}
        />
    ));

    const tableData = React.useMemo(() => {
        return filteredData
            .filter(row => row && typeof row === 'object' && row._id)
            .map(row => {
                const fileName = getFileNameOnly(row.name);
                
                return {
                    _id: row._id || '',
                    cells: [
                        <CheckboxCell 
                            key={row._id}
                            rowId={row._id}
                            isSelected={selectedSet.has(row._id)}
                            onSelect={() => handleCheckboxSelect(row)}
                        />,
                        <span 
                            key="filename" 
                            title={row.name || 'N/A'}
                            style={{ cursor: 'default' }}
                        >
                            {fileName || 'N/A'}
                        </span>,
                        row.type || 'N/A',
                        row.file_format || 'N/A',
                        row.created_at ? new Date(row.created_at).toLocaleString() : 'N/A'
                    ]
                };
            });
    }, [filteredData, selectedSet, handleCheckboxSelect]);

    const selectAllChecked = filteredData.length > 0 && filteredData.every(row => selected.includes(row._id));
    const selectAllIndeterminate = filteredData.length > 0 && filteredData.some(row => selected.includes(row._id)) && !selectAllChecked;

    return (
        <Modal isOpen={open} onClose={onClose} title="Dataset Details" titleIcon={titleIcon} className={styles.wideModal}>
            {loading && <Loading />}
            {error && <ErrorMessage message={error} />}
            {data && typeof data === 'object' && data.name && data.data_list && Array.isArray(data.data_list) && (
                <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    height: '100%', 
                    gap: '16px'
                }}>
                    <div className={styles.detailInfo} style={{ borderRadius: 12, background: '#f8f9fb', padding: 16, boxShadow: '0 1px 4px #0001', display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: 18 }}>{data.name}</div>
                        <div style={{ color: '#888', marginBottom: 4 }}>{data.description || <span style={{ color: '#bbb' }}>No description</span>}</div>
                        <div style={{ display: 'flex', gap: 16, fontSize: 14, alignItems: 'center' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Type:</b> {data.type || 'N/A'}</span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Total:</b> {data.total || 0}</span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><b>Created:</b> {(() => {
                                if (data && data.created_at) {
                                    return new Date(data.created_at).toLocaleString();
                                } else if (dataset && dataset.created_at) {
                                    return new Date(dataset.created_at).toLocaleString();
                                } else {
                                    return 'N/A';
                                }
                            })()}</span>
                        </div>
                    </div>
                    
                    <form onSubmit={handleUpload} style={{ display: 'flex', alignItems: 'flex-end', gap: 12, justifyContent: 'flex-end', flexShrink: 0 }}>
                        <FileUploadField
                            files={uploadFiles}
                            setFiles={updateUploadFiles}
                            fileError={uploadError}
                            setFileError={() => {}} // useDatasetData에서 처리
                            accept="*"
                            multiple={true}
                            allowFolders={true} // 폴더 선택 허용
                            uploadProgress={uploadProgress} // 배치 업로드 진행률
                        />
                        <Button
                            type="submit"
                            size="medium"
                            variant="primary"
                            disabled={uploading || !uploadFiles.length}
                            style={{ minWidth: 90 }}
                        >
                            {uploading ? <span style={{ display: 'inline-block', width: 16, height: 16, border: '2px solid #fff', borderTop: '2px solid #bfc6d1', borderRadius: '50%', animation: 'spin 1s linear infinite', verticalAlign: 'middle' }} /> : 'Upload'}
                        </Button>
                        <style>{`
                        @keyframes spin {
                          0% { transform: rotate(0deg); }
                          100% { transform: rotate(360deg); }
                        }
                        `}</style>
                    </form>

                    {/* 검색 및 선택 컨트롤 */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: '12px', flexShrink: 0 }}>
                        {/* 검색 영역 */}
                        <div style={{ position: 'relative', flex: 1, maxWidth: 400 }}>
                            <Search size={18} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', pointerEvents: 'none' }} />
                            <input
                                type="text"
                                placeholder="Search files..."
                                value={search}
                                onChange={e => setSearch(e.target.value)}
                                style={{
                                    width: '100%',
                                    padding: '10px 12px 10px 40px',
                                    border: '1.5px solid #d1d5db',
                                    borderRadius: '8px',
                                    fontSize: '14px',
                                    background: '#fff',
                                    fontWeight: '500',
                                    transition: 'all 0.2s ease',
                                    boxSizing: 'border-box'
                                }}
                            />
                            {search && (
                                <button
                                    type="button"
                                    onClick={() => setSearch('')}
                                    style={{
                                        position: 'absolute',
                                        right: 8,
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        background: 'none',
                                        border: 'none',
                                        color: '#9ca3af',
                                        cursor: 'pointer',
                                        padding: '4px',
                                        borderRadius: '4px',
                                        fontSize: '12px'
                                    }}
                                >
                                    Clear
                                </button>
                            )}
                        </div>

                        {/* 검색 결과 정보 */}
                        <div style={{ fontSize: '14px', color: '#6b7280', fontWeight: '500' }}>
                            {filteredData.length} of {data.data_list.length} files
                        </div>

                        {/* 액션 버튼들 */}
                        <div style={{ display: 'flex', gap: 8 }}>
                            <Button
                                size="medium"
                                variant="danger"
                                disabled={!selected.length || downloading}
                                onClick={toggleDeleteConfirm}
                            >
                                Delete Selected{selected.length > 0 ? ` (${selected.length})` : ''}
                            </Button>
                            <Button
                                size="medium"
                                variant="secondary"
                                disabled={!selected.length || downloading}
                                onClick={handleDownloadSelected}
                            >
                                <Download size={16} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                                {downloading ? 'Downloading...' : 'Download Selected'}
                            </Button>
                        </div>
                    </div>

                    {/* Select All 체크박스 */}
                    <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: '8px', flexShrink: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <input 
                                type="checkbox" 
                                checked={selectAllChecked}
                                indeterminate={selectAllIndeterminate || undefined}
                                onChange={handleSelectAll}
                                style={{ margin: 0, cursor: 'pointer' }}
                            />
                            <span style={{ fontSize: '12px', color: '#6b7280', cursor: 'pointer' }} onClick={handleSelectAll}>
                                Select All
                            </span>
                            {selected.length > 0 && (
                                <span style={{ fontSize: '12px', color: '#3b82f6' }}>
                                    ({selected.length} selected)
                                </span>
                            )}
                        </div>
                    </div>

                    {/* 테이블 영역 - 남은 공간을 모두 차지하고 스크롤 가능 */}
                    <div style={{ 
                        width: '100%', 
                        flex: 1, 
                        minHeight: 0,
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <Table
                            columns={[
                                // 체크박스 컬럼 헤더 (빈 문자열로 표시)
                                '',
                                'Name', 'Type', 'Format', 'Created'
                            ]}
                            data={tableData.map(row => row.cells)}
                            rowKey="_id"
                            onRowClick={(_, idx) => {
                                if (!filteredData || !Array.isArray(filteredData) || !filteredData[idx]) return;
                                handleSelect(filteredData[idx]);
                            }}
                            selectedId={null}
                            selectedRowClassName={styles.selectedRow}
                            maxHeight="100%"
                            virtualized={true}
                        />
                    </div>

                    {/* 빈 검색 결과 상태 */}
                    {filteredData.length === 0 && data.data_list.length > 0 && (
                        <div style={{ textAlign: 'center', padding: '40px 20px', color: '#6b7280', flexShrink: 0 }}>
                            <p style={{ margin: 0, fontSize: '14px', fontWeight: '500' }}>
                                {search ? 'No files found matching your search.' : 'No files available.'}
                            </p>
                        </div>
                    )}

                    {showDeleteConfirm && (
                        <Modal isOpen={showDeleteConfirm} onClose={toggleDeleteConfirm} title="Delete Data" className={styles.confirmModal}>
                            <div style={{ padding: 16, fontSize: 16, color: '#d32f2f', textAlign: 'center' }}>
                                <Trash2 size={32} style={{ marginBottom: 8 }} />
                                <div>Are you sure you want to delete the selected data?</div>
                                <div style={{ fontSize: 14, color: '#888', marginTop: 8 }}>This action cannot be undone.</div>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginTop: 16 }}>
                                <Button variant="secondary" onClick={toggleDeleteConfirm} size="medium">Cancel</Button>
                                <Button variant="danger" onClick={handleDelete} size="medium">Delete</Button>
                            </div>
                        </Modal>
                    )}
                </div>
            )}
        </Modal>
    );
};

export default DatasetDataPanel; 