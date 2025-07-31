import React from 'react';
import SectionTitle from "../components/common/SectionTitle.jsx";

const ServiceProcessPage = () => {
  return (
    <div>
      <SectionTitle children="Service Process" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Service Process Page</h3>
        <p>각 기능 별 handler 관리가 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default ServiceProcessPage;
