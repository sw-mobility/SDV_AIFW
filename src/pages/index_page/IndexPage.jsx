import React, { useState } from 'react';
import Header from '../../components/layout/Header.jsx';
import Footer from '../../components/layout/Footer.jsx';
import TabNavigation from '../../components/common/TabNavigation.jsx';
import styles from '../../components/layout/Layout.module.css';
import pageStyles from './IndexPage.module.css';
import ProjectsTab from './index_tab/ProjectsTab.jsx';
import DatasetsTab from './index_tab/DatasetsTab.jsx';

const IndexPage = ({ mockState }) => {
    const [activeTab, setActiveTab] = useState('projects'); // 'projects' or 'data'

    // 탭 설정
    const tabs = [
        { value: 'projects', label: 'Projects' },
        { value: 'data', label: 'Data Management' }
    ];

    return (
        <div className={styles['main-layout']}>
            <Header />
            <div className={styles['main-body']}>
                <main className={styles['main-content']}>
                    <div className={pageStyles.container}>
                        <TabNavigation
                            tabs={tabs}
                            activeTab={activeTab}
                            onTabChange={setActiveTab}
                        />

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