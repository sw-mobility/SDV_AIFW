import React from 'react';
import {BrowserRouter as Router, Routes, Route, Navigate} from 'react-router-dom';

import MainLayout from '../components/layout/MainLayout';
import IndexPage from '../pages/IndexPage';

const AppRoutes = () => (
    <Router>
        <Routes>
            <Route path="/" element={<IndexPage/>}/>

            <Route path="/projects/:projectId" element={<MainLayout/>}>
                <Route index element={<ProjectHomePage/>}/>
                <Route path="training" element={<TrainingPage/>}/>
                <Route path="labelling" element={<LabellingPage/>}/>
                <Route path="optimization" element={<OptimizationPage/>}/>
                <Route path="validation" element={<ValidationPage/>}/>
                <Route path="data-management" element={<DataManagementPage/>}/>
                <Route path="data-acquisition" element={<DataAcquisitionPage/>}/>
                <Route path="deployment" element={<DeploymentPage/>}/>
                <Route path="service-process" element={<ServiceProcessPage/>}/>
            </Route>

            <Route path="*" element={<Navigate to="/" replace/>}/>
        </Routes>
    </Router>
);

export default AppRoutes;