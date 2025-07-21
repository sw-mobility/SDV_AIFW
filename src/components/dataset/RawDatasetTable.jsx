import React, { useState, useMemo } from 'react';
import Table from '../common/Table.jsx';
import { Search } from 'lucide-react';
import pageStyles from '../../pages/labeling_page/LabelingPage.module.css';
import tableStyles from './RawDatasetTable.module.css';

export default function RawDatasetTable({ columns, data, onRowClick, selectedId, searchPlaceholder = 'Search...' }) {
  const [search, setSearch] = useState('');
  const filtered = useMemo(() => {
    if (!search) return data;
    return data.filter(row => row.name?.toLowerCase().includes(search.toLowerCase()));
  }, [data, search]);

  const tableData = filtered.map(row => ({
    id: row.id,
    cell: (
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
        <span style={{fontWeight: 600, color: '#1e293b'}}>{row.name}</span>
        <span style={{fontSize: 12, color: '#b6c2d6', marginTop: 2}}>
          {row.createdAt ? row.createdAt : (row.lastModified || '-')}
        </span>
      </div>
    )
  }));

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
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
      <Table
        columns={["Name"]}
        data={tableData}
        onRowClick={(_, idx) => onRowClick && onRowClick(filtered[idx])}
        selectedId={selectedId}
        rowKey="id"
        selectedRowClassName={tableStyles.selectedRow}
      />
    </div>
  );
}