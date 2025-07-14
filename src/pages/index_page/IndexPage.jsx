import React, { useState } from 'react';
import Header from '../../components/layout/Header.jsx';
import Footer from '../../components/layout/Footer.jsx';
import styles from '../../components/layout/Layout.module.css';
import pageStyles from './IndexPage.module.css';
import ProjectsTab from './ProjectsTab.jsx';
import DatasetsTab from './DatasetsTab.jsx';

const IndexPage = ({ mockState }) => {
    const [activeTab, setActiveTab] = useState('projects'); // 'projects' or 'data'



    return (
        <div className={styles['main-layout']}>
            <Header />
            <div className={styles['main-body']}>
                <main className={styles['main-content']}>
                    <div className={pageStyles.container}>
                        <div className={pageStyles.tabNavigation}>
                            <button
                                className={`${pageStyles.tabButton} ${activeTab === 'projects' ? pageStyles.activeTab : ''}`}
                                onClick={() => setActiveTab('projects')}
                            >
                                Projects
                            </button>
                            <button
                                className={`${pageStyles.tabButton} ${activeTab === 'data' ? pageStyles.activeTab : ''}`}
                                onClick={() => setActiveTab('data')}
                            >
                                Data Management
                            </button>
                        </div>

                        {activeTab === 'projects' ? (
                            <ProjectsTab mockState={mockState} />
                        ) : (
                            <DatasetsTab mockState={mockState} />
                        )}
                    </div>
                </main>
            </div>
            <Footer />
        </div>
    );
};

export default IndexPage;