import React, { useState } from 'react';
import Header from '../../components/layout/Header.jsx';
import Footer from '../../components/layout/Footer.jsx';
import styles from '../../components/layout/Layout.module.css';
import pageStyles from './IndexPage.module.css';
import Card, { CardGrid } from "../../components/ui/Card.jsx";
import SectionTitle from "../../components/ui/SectionTitle.jsx";
import CreateProjectModal from "./CreateProjectModal.jsx";
import { Plus, FolderOpen, Calendar, ChevronDown, ChevronUp } from 'lucide-react';
import Chip from '@mui/material/Chip';

/**
 *
 * @returns {Element}
 * @constructor
 */
const IndexPage = () => {
    const [showMore, setShowMore] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const cardsPerPage = 8;
    const [projects, setProjects] = useState([
        { id: 1, name: 'AI Image Recognition', status: 'Active', lastModified: '2024-01-15' },
        { id: 2, name: 'Natural Language Processing', status: 'Active', lastModified: '2024-01-14' },
        { id: 3, name: 'Predictive Analytics', status: 'Active', lastModified: '2024-01-13' },
        { id: 4, name: 'Computer Vision Model', status: 'Training', lastModified: '2024-01-12' },
        { id: 5, name: 'Recommendation System', status: 'Active', lastModified: '2024-01-11' },
        { id: 6, name: 'Fraud Detection AI', status: 'Deployed', lastModified: '2024-01-10' },
        { id: 7, name: 'Sentiment Analysis', status: 'Active', lastModified: '2024-01-09' },
        { id: 8, name: 'Object Detection', status: 'Training', lastModified: '2024-01-08' },
        { id: 9, name: 'Text Classification', status: 'Active', lastModified: '2024-01-07' },
        { id: 10, name: 'Speech Recognition', status: 'Deployed', lastModified: '2024-01-06' },
        { id: 11, name: 'Data Mining Project', status: 'Active', lastModified: '2024-01-05' },
        { id: 12, name: 'Neural Network Model', status: 'Training', lastModified: '2024-01-04' },
    ]);

    const handleCreateProject = () => {
        setIsModalOpen(true);
    };

    const handleModalClose = () => {
        setIsModalOpen(false);
    };

    const handleCreateProjectSubmit = (projectName) => {
        const newProject = {
            id: Date.now(),
            name: projectName,
            status: 'Active',
            lastModified: new Date().toISOString().slice(0, 10),
        };
        setProjects(prev => [newProject, ...prev]);
        setIsModalOpen(false);
        window.location.href = `/projects/${newProject.id}`;
    };

    const handleProjectClick = (projectId) => {
        window.location.href = `/projects/${projectId}`;
    };

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'Active': return 'success';
            case 'Training': return 'warning';
            case 'Deployed': return 'info';
            default: return 'default';
        }
    };

    const getStatusText = (status) => {
        switch (status) {
            case 'Active': return 'active';
            case 'Training': return 'training';
            case 'Deployed': return 'deployed';
            default: return status;
        }
    };

    const CreateProjectCard = () => (
        <Card onClick={handleCreateProject} className={pageStyles.createProjectCard}>
            <div className={pageStyles.createCard}>
                <Plus size={32} className={pageStyles.createCardIcon} />
                <div className={pageStyles.createCardText}>
                    Create New Project
                </div>
            </div>
        </Card>
    );

    const ProjectCard = ({ project }) => (
        <Card onClick={() => handleProjectClick(project.id)}>
            <div className={pageStyles.projectCard}>
                <Chip
                    label={getStatusText(project.status)}
                    color={getStatusColor(project.status)}
                    size="small"
                    variant="outlined"
                    style={{
                        position: 'absolute',
                        top: '12px',
                        right: '12px',
                        fontSize: '11px',
                        height: '20px'
                    }}
                />

                <div className={pageStyles.projectIcon}>
                    <FolderOpen size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={pageStyles.projectName}>
                    {project.name}
                </div>

                <div className={pageStyles.projectDate}>
                    <Calendar size={14} />
                    {project.lastModified}
                </div>
            </div>
        </Card>
    );

    const allCards = [<CreateProjectCard key="create" />, ...projects.map(project => (
        <ProjectCard key={project.id} project={project} />
    ))];
    const visibleCards = showMore ? allCards : allCards.slice(0, cardsPerPage);

    return (
        <div className={styles['main-layout']}>
            <Header />
            <SectionTitle children="Projects" size='md' />
            <div className={styles['main-body']}>
                <main className={styles['main-content']}>
                    <div className={pageStyles.container}>
                        <CardGrid columns={4} gap="2rem">
                            {visibleCards}
                        </CardGrid>

                        {allCards.length > cardsPerPage && (
                            <div className={pageStyles.loadMoreContainer}>
                                <button
                                    onClick={handleToggleShowMore}
                                    className={pageStyles.moreButton}
                                >
                                    <span className={pageStyles.moreText}>
                                        {showMore ? 'Show Less' : `Show ${allCards.length - cardsPerPage} More`}
                                    </span>
                                    <div className={`${pageStyles.chevron} ${showMore ? pageStyles.chevronUp : ''}`}>
                                        <ChevronDown size={14} />
                                    </div>
                                </button>
                            </div>
                        )}
                    </div>
                </main>
            </div>
            <Footer />

            <CreateProjectModal
                isOpen={isModalOpen}
                onClose={handleModalClose}
                onSubmit={handleCreateProjectSubmit}
            />
        </div>
    );
};

export default IndexPage;