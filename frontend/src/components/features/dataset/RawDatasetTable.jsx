import React, { useState, useMemo, useCallback } from 'react';
import Table from '../../ui/atoms/Table.jsx';
import { Search, ChevronLeft, ChevronRight } from 'lucide-react';
import pageStyles from '../../../pages/labeling_page/LabelingPage.module.css';
import tableStyles from './RawDatasetTable.module.css';

/**
 * Raw 데이터셋을 테이블 형태로 표시
 * labeling page 등에서 raw dataset 표 표출 필요할 때 사용
 * 주요 기능:
 * 검색 기능
 * 데이터셋 선택
 * 이름과 생성일 표시
 * 대용량 데이터셋을 위한 페이지네이션
 * @param data
 * @param onRowClick
 * @param selectedId
 * @param searchPlaceholder
 * @param itemsPerPage
 * @returns {Element}
 * @constructor
 */
export default function RawDatasetTable({data, onRowClick, selectedId, searchPlaceholder = 'Search...', itemsPerPage = 50 }) {
  const [search, setSearch] = useState('');
  const [currentPage, setCurrentPage] = useState(0);

  // 검색된 데이터 (최신 순으로 정렬)
  const filteredData = useMemo(() => {
    let filtered = data;
    if (search) {
      filtered = data.filter(row => 
        row.name?.toLowerCase().includes(search.toLowerCase()) ||
        row.createdAt?.toLowerCase().includes(search.toLowerCase())
      );
    }
    
    // 최신 순으로 정렬 (createdAt 기준)
    return filtered.sort((a, b) => {
      const dateA = new Date(a.createdAt || a.created_at || 0);
      const dateB = new Date(b.createdAt || b.created_at || 0);
      return dateB - dateA; // 내림차순 (최신이 위로)
    });
  }, [data, search]);

  // 페이지네이션
  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const startIndex = currentPage * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedData = filteredData.slice(startIndex, endIndex);

  // 검색이 변경되면 첫 페이지로 이동
  React.useEffect(() => {
    setCurrentPage(0);
  }, [search]);

  const tableData = paginatedData.map(row => ({
    id: row._id || row.id,
    cells: [
      <div key="name-cell" style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
        <span style={{fontWeight: 600, color: '#1e293b'}}>{row.name}</span>
        <span style={{fontSize: 12, color: '#b6c2d6', marginTop: 2}}>
          {row.createdAt ? row.createdAt : (row.lastModified || '-')}
        </span>
      </div>
    ]
  }));

  const handlePageChange = useCallback((newPage) => {
    setCurrentPage(Math.max(0, Math.min(newPage, totalPages - 1)));
  }, [totalPages]);

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      {/* 검색 및 정보 영역 */}
      <div className={tableStyles.headerSection}>
        <div className={pageStyles.searchWrap}>
          <Search size={18} className={pageStyles.searchIcon} />
          <input
            type="text"
            placeholder={searchPlaceholder}
            value={search}
            onChange={e => setSearch(e.target.value)}
            className={pageStyles.searchInput}
          />
        </div>
        
        {/* 검색 결과 정보 */}
        <div className={tableStyles.searchInfo}>
          <span className={tableStyles.resultCount}>
            {filteredData.length} of {data.length} datasets
          </span>
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              className={tableStyles.clearSearchBtn}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* 테이블 */}
      <div className={tableStyles.tableContainer}>
        <Table
          columns={["Name"]}
          data={tableData}
          onRowClick={(_, idx) => onRowClick && onRowClick(paginatedData[idx])}
          selectedId={selectedId}
          rowKey="id"
          selectedRowClassName={tableStyles.selectedRow}
          virtualized={true} // 가상화 활성화
        />
      </div>

      {/* 페이지네이션 */}
      {totalPages > 1 && (
        <div className={tableStyles.pagination}>
          <div className={tableStyles.paginationInfo}>
            <span>
              Showing {startIndex + 1}-{Math.min(endIndex, filteredData.length)} of {filteredData.length} results
            </span>
          </div>
          
          <div className={tableStyles.paginationControls}>
            <button
              type="button"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 0}
              className={tableStyles.pageBtn}
            >
              <ChevronLeft size={16} />
              Previous
            </button>
            
            <div className={tableStyles.pageNumbers}>
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i;
                } else if (currentPage < 3) {
                  pageNum = i;
                } else if (currentPage >= totalPages - 3) {
                  pageNum = totalPages - 5 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                
                return (
                  <button
                    key={pageNum}
                    type="button"
                    onClick={() => handlePageChange(pageNum)}
                    className={`${tableStyles.pageNumber} ${currentPage === pageNum ? tableStyles.activePage : ''}`}
                  >
                    {pageNum + 1}
                  </button>
                );
              })}
            </div>
            
            <button
              type="button"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages - 1}
              className={tableStyles.pageBtn}
            >
              Next
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* 빈 결과 상태 */}
      {filteredData.length === 0 && (
        <div className={tableStyles.emptyState}>
          <p>{search ? 'No datasets found matching your search.' : 'No datasets available.'}</p>
        </div>
      )}
    </div>
  );
}