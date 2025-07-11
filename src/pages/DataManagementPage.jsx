import React from 'react';
import SectionTitle from "../components/ui/SectionTitle.jsx";

const DataManagementPage = () => {
  return (
    <div>
      <SectionTitle children="Data Management" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Data Management Page</h3>
        <p>dataset 목록, 데이터 관리 기능, 다른 프로젝트 dataset 보기, 수정된 파일 관리가 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default DataManagementPage;
