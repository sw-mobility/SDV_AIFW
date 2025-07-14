import React from 'react';
import SectionTitle from "../components/ui/SectionTitle.jsx";

const TrainingPage = () => {
  return (
    <div>
      <SectionTitle children="Training" size="lg" />
      <div style={{ padding: '2rem' }}>
        <h3>Training Page</h3>
        <p>모든 구분 기능이 여기에 구현됩니다.</p>
      </div>
    </div>
  );
};

export default TrainingPage;
