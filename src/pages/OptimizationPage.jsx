import React from 'react';
import SectionTitle from "../components/ui/SectionTitle.jsx";

const OptimizationPage = () => {
  return (
    <div>
      <SectionTitle children="Optimization" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Optimization Page</h3>
        <p>target 보드 선택, model 선택, test dataset 선택, 옵션 변경, progress bar가 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default OptimizationPage;
