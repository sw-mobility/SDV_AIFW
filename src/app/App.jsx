import './App.css'
import { useEffect } from 'react';
import AppRoutes from './routes/AppRoutes.jsx';
import { initializeApp } from '../api/init.js';

function App() {
    useEffect(() => {
        // 앱 시작 시 초기화 API 호출
        initializeApp().catch(error => {
            console.error('App initialization failed:', error);
        });
    }, []);

    return (
        <div className="app">
            <AppRoutes />
        </div>
    )
}

export default App
