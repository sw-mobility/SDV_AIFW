import React from 'react';
import SectionTitle from "../components/ui/SectionTitle.jsx";

const ValidationPage = () => {
  return (
    <div>
      <SectionTitle children="Validation" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Validation Page</h3>
        <p>model 선택, metric 선택, test dataset 선택, 결과 저장 → DB 기록이 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default ValidationPage;



