import { useState } from 'react';

/**
 * Index 페이지의 탭 관리를 위한 커스텀 훅
 * 
 * @returns {Object} 탭 관련 상태와 핸들러
 */
export const useIndexTabs = () => {
    const [activeTab, setActiveTab] = useState('projects');

    const tabs = [
        { value: 'projects', label: 'Projects' },
        { value: 'data', label: 'Data Management' }
    ];

    const handleTabChange = (tabValue) => {
        setActiveTab(tabValue);
    };

    return {
        activeTab,
        tabs,
        handleTabChange
    };
}; 