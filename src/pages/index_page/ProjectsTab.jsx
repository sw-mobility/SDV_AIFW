import React, { useState, useEffect } from 'react';
import {Plus, FolderOpen, ChevronDown, Trash2, Pencil} from 'lucide-react';
import Card, { CardGrid } from '../../components/ui/Card.jsx';
import CreateModal from './CreateModal.jsx';
import styles from './IndexPage.module.css';
import Chip from '@mui/material/Chip';
import { Calendar } from 'lucide-react';
import { fetchProjects, createProject, deleteProject, updateProject } from '../../api/projects.js';

const ProjectsTab = ({ mockState }) => {
    const [showMore, setShowMore] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editModalOpen, setEditModalOpen] = useState(false);
    const [editProject, setEditProject] = useState(null);
    const cardsPerPage = 8;

    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        fetchProjects(mockState)
            .then(res => setProjects(res.data))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    }, [mockState]);

    if (loading || mockState?.loading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>;
    if (error || mockState?.error) return <div style={{ color: 'red', padding: '2rem', textAlign: 'center' }}>Error: {error || 'Mock error!'}</div>;
    if (projects.length === 0 || mockState?.empty) return <div style={{ padding: '2rem', textAlign: 'center' }}>No projects found.</div>;

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

    // 프로젝트 목록 새로고침
    const reloadProjects = () => {
        setLoading(true);
        setError(null);
        fetchProjects(mockState)
            .then(res => setProjects(res.data))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    };

    //삭제
    const handleDeleteProject = async (projectId) => {
        setLoading(true);
        setError(null);
        try {
            await deleteProject(projectId);
            reloadProjects();
        } catch (e) {
            setError(e.message);
            setLoading(false);
        }
    };

    // 편집 모달 열기
    const handleEditProject = (project) => {
        setEditProject(project);
        setEditModalOpen(true);
    };
    // 편집 모달 닫기
    const handleEditModalClose = () => {
        setEditModalOpen(false);
        setEditProject(null);
    };
    // 편집 저장
    const handleEditProjectSubmit = async (newName) => {
        setLoading(true);
        setError(null);
        try {
            await updateProject(editProject.id, { name: newName });
            reloadProjects();
            setEditModalOpen(false);
            setEditProject(null);
        } catch (e) {
            setError(e.message);
            setLoading(false);
        }
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
                        title="Edit"
                        onClick={e => { e.stopPropagation(); handleEditProject(project); }}
                        style={{ marginRight: 8 }}
                    >
                        <Pencil size={14} />
                    </button>
                    <button 
                        className={styles.actionButton} 
                        title="Delete"
                        onClick={e => { e.stopPropagation(); handleDeleteProject(project.id); }}
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
            <CreateModal
                isOpen={editModalOpen}
                onClose={handleEditModalClose}
                onSubmit={handleEditProjectSubmit}
                initialName={editProject?.name || ''}
                title="Edit Project Name"
                submitLabel="Save"
            />
        </>
    );
};

export default ProjectsTab; 