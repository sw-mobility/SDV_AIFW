import React from 'react';
import {BrowserRouter as Router, Routes, Route, Navigate, Outlet} from 'react-router-dom';

import IndexPage from '../../pages/index_page/IndexPage.jsx';
import ProjectHomePage from '../../pages/project_home/ProjectHomePage.jsx';
import TrainingPage from '../../pages/training_page/TrainingPage.jsx';
import LabelingPage from '../../pages/labeling_page/LabelingPage.jsx';
import OptimizationPage from '../../pages/optimization_page/OptimizationPage.jsx';
import ValidationPage from '../../pages/validation_page/ValidationPage.jsx';
import DeploymentPage from '../../pages/DeploymentPage.jsx';
import ServiceProcessPage from '../../pages/ServiceProcessPage.jsx';
import MainLayout from '../../components/layout/MainLayout.jsx';

const IdePage = () => (<div style={{
        padding: 40,
        fontSize: 24,
        textAlign: 'center',
        color: 'var(--color-text-main)',
        fontFamily: 'var(--font-family)'
    }}>
        [ Training IDE ]
    </div>);

const LabelingDetailPage = () => (<div style={{
        padding: 40,
        fontSize: 24,
        textAlign: 'center',
        color: 'var(--color-text-main)',
        fontFamily: 'var(--font-family)'
    }}>
        [ Labeling Detail ]
    </div>);

const AppRoutes = () => (<Router>
        <Routes>
            <Route path="/" element={<IndexPage/>}/>
            <Route path="/projects/:projectName" element={<MainLayout/>}>
                <Route index element={<ProjectHomePage/>}/>
                <Route path="training" element={<TrainingPage/>}/>
                <Route path="labeling" element={<LabelingPage/>}/>
                <Route path="optimization" element={<OptimizationPage/>}/>
                <Route path="validation" element={<ValidationPage/>}/>
                <Route path="deployment" element={<DeploymentPage/>}/>
                <Route path="service-process" element={<ServiceProcessPage/>}/>
            </Route>


            <Route path="*" element={<Navigate to="/" replace/>}/>
        </Routes>
    </Router>);

export default AppRoutes;