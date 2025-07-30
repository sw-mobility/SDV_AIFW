import './App.css'
import AppRoutes from './routes/AppRoutes.jsx';
import { DatasetProvider } from './context/DatasetContext.jsx';

function App() {
    return (
        <div className="app">
            <DatasetProvider>
                <AppRoutes />
            </DatasetProvider>
        </div>
    )
}

export default App
