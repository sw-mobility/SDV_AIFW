import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';

import IndexPage from '../pages/index_page/IndexPage.jsx';
import ProjectHomePage from '../pages/ProjectHomePage';
import TrainingPage from '../pages/TrainingPage';
import LabellingPage from '../pages/LabellingPage';
import OptimizationPage from '../pages/OptimizationPage';
import ValidationPage from '../pages/ValidationPage';
import DataManagementPage from '../pages/DataManagementPage';
import DataAcquisitionPage from '../pages/DataAcquisitionPage';
import DeploymentPage from '../pages/DeploymentPage';
import ServiceProcessPage from '../pages/ServiceProcessPage';

const IdePage = () => (
  <div style={{ 
    padding: 40, 
    fontSize: 24, 
    textAlign: 'center',
    color: 'var(--color-text-main)',
    fontFamily: 'var(--font-family)'
  }}>
    [ Training IDE ]
  </div>
);

const LabellingDetailPage = () => (
  <div style={{ 
    padding: 40, 
    fontSize: 24, 
    textAlign: 'center',
    color: 'var(--color-text-main)',
    fontFamily: 'var(--font-family)'
  }}>
    [ Labelling Detail ]
  </div>
);

const AppRoutes = () => (
  <Router>
    <Routes>
      <Route path="/" element={<IndexPage />} />
      
      <Route path="/projects/:projectId" element={<Outlet />}>
        <Route index element={<ProjectHomePage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="training/ide" element={<IdePage />} />
        <Route path="labelling" element={<LabellingPage />} />
        <Route path="labelling/:datasetId" element={<LabellingDetailPage />} />
        <Route path="optimization" element={<OptimizationPage />} />
        <Route path="validation" element={<ValidationPage />} />
        <Route path="data-management" element={<DataManagementPage />} />
        <Route path="data-acquisition" element={<DataAcquisitionPage />} />
      </Route>
      
      <Route path="/deployment" element={<DeploymentPage />} />
      <Route path="/service-process" element={<ServiceProcessPage />} />
      
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </Router>
);

export default AppRoutes;