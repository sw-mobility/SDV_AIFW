import { BrowserRouter, Routes, Route, Outlet } from 'react-router-dom';

const Placeholder = ({ name }) => <div style={{ padding: 40, fontSize: 24 }}>[ {name} ]</div>;

function NotFound() {
  return null;
}

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Placeholder name="Home" />} />
        <Route path="/projects/:projectId" element={<Outlet />}>
          <Route index element={<Placeholder name="Project Default" />} />
          <Route path="training" element={<Placeholder name="Training" />} />
          <Route path="training/ide" element={<Placeholder name="Training IDE" />} />
          <Route path="labelling" element={<Placeholder name="Labelling" />} />
          <Route path="labelling/:datasetId" element={<Placeholder name="Labelling Detail" />} />
          <Route path="optimization" element={<Placeholder name="Optimization" />} />
          <Route path="validation" element={<Placeholder name="Validation" />} />
          <Route path="data-management" element={<Placeholder name="Data Management" />} />
          <Route path="data-acquisition" element={<Placeholder name="Data Acquisition" />} />
        </Route>
        <Route path="/deployment" element={<Placeholder name="Deployment" />} />
        <Route path="/service-process" element={<Placeholder name="Service Process" />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
} 