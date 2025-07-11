import React, { useState } from 'react';
import { Plus, FolderOpen, ChevronDown, Trash2 } from 'lucide-react';
import { useLocalStorageState } from '../../hooks/useLocalStorageState.js';
import Card, { CardGrid } from '../../components/ui/Card.jsx';
import CreateModal from './CreateModal.jsx';
import styles from './IndexPage.module.css';
import Chip from '@mui/material/Chip';
import { Calendar } from 'lucide-react';

const ProjectsTab = () => {
    const [showMore, setShowMore] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const cardsPerPage = 8;

    const [projects, setProjects] = useLocalStorageState('projects', [
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

    const handleDeleteProject = (projectId) => {
        setProjects(prev => prev.filter(project => project.id !== projectId));
    };

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const CreateProjectCard = () => (
        <Card onClick={handleCreateProject} className={styles.createCard}>
            <div className={styles.createCardContent}>
                <Plus size={32} className={styles.createCardIcon} />
                <div className={styles.createCardText}>
                    Create New Project
                </div>
            </div>
        </Card>
    );

    const getStatusColor = (status) => {
        switch (status) {
            case 'Active': return 'success';
            case 'Training': return 'warning';
            case 'Deployed': return 'info';
            case 'Processing': return 'warning';
            default: return 'default';
        }
    };

    const getStatusText = (status) => {
        switch (status) {
            case 'Active': return 'active';
            case 'Training': return 'training';
            case 'Deployed': return 'deployed';
            case 'Processing': return 'processing';
            default: return status;
        }
    };

    const ProjectCard = ({ project }) => (
        <Card onClick={() => handleProjectClick(project.id)} className={styles.projectCard}>
            <div className={styles.cardContent}>
                <Chip
                    label={getStatusText(project.status)}
                    color={getStatusColor(project.status)}
                    size="small"
                    variant="outlined"
                    className={styles.statusChip}
                />

                <div className={styles.cardIcon}>
                    <FolderOpen size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {project.name}
                </div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {project.lastModified}
                </div>

                <div className={styles.cardActions}>
                    <button 
                        className={styles.actionButton} 
                        title="Delete"
                        onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteProject(project.id);
                        }}
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const allProjectCards = [
        <CreateProjectCard key="create" />,
        ...projects.map(project => (
            <ProjectCard key={project.id} project={project} />
        ))
    ];
    
    const visibleProjectCards = showMore ? allProjectCards : allProjectCards.slice(0, cardsPerPage);

    return (
        <>
            <CardGrid gap="2rem">
                {visibleProjectCards}
            </CardGrid>

            {allProjectCards.length > cardsPerPage && (
                <div className={styles.loadMoreContainer}>
                    <button
                        onClick={handleToggleShowMore}
                        className={styles.moreButton}
                    >
                        <span className={styles.moreText}>
                            {showMore ? 'Show Less' : `Show ${allProjectCards.length - cardsPerPage} More`}
                        </span>
                        <div className={`${styles.chevron} ${showMore ? styles.chevronUp : ''}`}>
                            <ChevronDown size={14} />
                        </div>
                    </button>
                </div>
            )}

            <CreateModal
                isOpen={isModalOpen}
                onClose={handleModalClose}
                onSubmit={handleCreateProjectSubmit}
            />
        </>
    );
};

export default ProjectsTab; 