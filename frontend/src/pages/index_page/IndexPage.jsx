import React from 'react';
import Header from '../../components/layout/Header.jsx';
import Footer from '../../components/layout/Footer.jsx';
import TabNavigation from '../../components/ui/atoms/TabNavigation.jsx';
import styles from '../../components/layout/Layout.module.css';
import pageStyles from './IndexPage.module.css';
import ProjectsTab from './index_tab/ProjectsTab.jsx';
import DatasetsTab from './index_tab/DatasetsTab.jsx';
import { useIndexTabs } from '../../hooks/index.js';

const IndexPage = () => {
    const { activeTab, tabs, handleTabChange } = useIndexTabs();

    return (
        <div className={styles['main-layout']}>
            <Header />
            <div className={styles['main-body']}>
                <main className={styles['main-content']}>
                    <div className={pageStyles.container}>
                        <TabNavigation
                            tabs={tabs}
                            activeTab={activeTab}
                            onTabChange={handleTabChange}
                        />

                        <div className={pageStyles.tabContent}>
                            <div className={pageStyles.tabPanel} style={{ display: activeTab === 'projects' ? 'block' : 'none' }}>
                                <ProjectsTab />
                            </div>
                            <div className={pageStyles.tabPanel} style={{ display: activeTab === 'data' ? 'block' : 'none' }}>
                                <DatasetsTab />
                            </div>
                        </div>
                    </div>
                </main>
            </div>
            <Footer />
        </div>
    );
};

export default IndexPage;