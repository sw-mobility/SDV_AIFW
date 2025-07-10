import './App.css'
import Header from './components/layout/Header.jsx';
import Footer from "./components/layout/Footer.jsx";
import CodeEditor from "./components/ui/CodeEditor.jsx";
import Sidebar from "./components/layout/Sidebar.jsx";

function App() {
  return (
    <div className="app">
        <Header />
        <CodeEditor/>
        <Footer />
    </div>
  )
}

export default App
