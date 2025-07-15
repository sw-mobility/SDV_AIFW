import React from 'react';
import {BrowserRouter as Router, Routes, Route, Navigate, Outlet} from 'react-router-dom';

import IndexPage from '../pages/index_page/IndexPage.jsx';
import ProjectHomePage from '../pages/ProjectHomePage';
import TrainingPage from '../pages/training_page/TrainingPage.jsx';
import LabelingPage from '../pages/LabelingPage.jsx';
import OptimizationPage from '../pages/OptimizationPage';
import ValidationPage from '../pages/ValidationPage';
import DeploymentPage from '../pages/DeploymentPage';
import ServiceProcessPage from '../pages/ServiceProcessPage';
import MainLayout from '../components/layout/MainLayout';

const IdePage = () => (<div style={{
        padding: 40,
        fontSize: 24,
        textAlign: 'center',
        color: 'var(--color-text-main)',
        fontFamily: 'var(--font-family)'
    }}>
        [ Training IDE ]
    </div>);

const LabellingDetailPage = () => (<div style={{
        padding: 40,
        fontSize: 24,
        textAlign: 'center',
        color: 'var(--color-text-main)',
        fontFamily: 'var(--font-family)'
    }}>
        [ Labelling Detail ]
    </div>);

const AppRoutes = () => (<Router>
        <Routes>
            <Route path="/" element={<IndexPage/>}/>
            <Route path="/projects/:projectId" element={<MainLayout/>}>
                <Route index element={<ProjectHomePage/>}/>
                <Route path="training" element={<TrainingPage/>}/>
                <Route path="training/ide" element={<IdePage/>}/>
                <Route path="labelling" element={<LabelingPage/>}/>
                <Route path="labelling/:datasetId" element={<LabellingDetailPage/>}/>
                <Route path="optimization" element={<OptimizationPage/>}/>
                <Route path="validation" element={<ValidationPage/>}/>
                <Route path="deployment" element={<DeploymentPage/>}/>
                <Route path="service-process" element={<ServiceProcessPage/>}/>
            </Route>


            <Route path="*" element={<Navigate to="/" replace/>}/>
        </Routes>
    </Router>);

export default AppRoutes;