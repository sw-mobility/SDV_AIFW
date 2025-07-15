import React from 'react';
import SectionTitle from "../components/common/SectionTitle.jsx";

const LabelingPage = () => {
  return (
    <div>
      <SectionTitle children="Labeling" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Labeling Page</h3>
        <p>데이터셋 목록, raw 데이터 목록, 레이블 옵션/설명, labeling progress, 완료 labeling sample이 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default LabelingPage;
