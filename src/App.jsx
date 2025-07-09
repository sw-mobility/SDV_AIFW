import './App.css'
import CodeEditor from './components/ui/CodeEditor.jsx';
import Header from './components/layout/Header.jsx';
import Footer from "./components/layout/Footer.jsx";
import Sidebar from "./components/layout/Sidebar.jsx";
import Button from "./components/ui/Button.jsx";
import Select from "./components/ui/Select.jsx";
import Table from "./components/ui/Table.jsx";

function App() {
  return (
    <div className="app">
        <Header />
        {/*<Sidebar />*/}
        <CodeEditor />
        <Footer />
    </div>
  )
}

export default App
