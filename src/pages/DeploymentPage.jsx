import React from 'react';
import SectionTitle from "../components/common/SectionTitle.jsx";

const DeploymentPage = () => {
  return (
    <div>
      <SectionTitle children="Deployment" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Deployment Page</h3>
        <p>모델 배포 관리가 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default DeploymentPage;
