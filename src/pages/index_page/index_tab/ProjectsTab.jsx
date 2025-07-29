import React, { useState, useEffect } from 'react';
import {Plus, FolderOpen, ChevronDown, Trash2, Pencil} from 'lucide-react';
import Card, { CardGrid } from '../../../components/common/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar } from 'lucide-react';
import { fetchProjects, createProject, deleteProject, updateProject } from '../../../api/projects.js';
import StatusChip from '../../../components/common/StatusChip.jsx';
import Loading from '../../../components/common/Loading.jsx';
import ErrorMessage from '../../../components/common/ErrorMessage.jsx';
import EmptyState from '../../../components/common/EmptyState.jsx';
import ShowMoreGrid from '../../../components/common/ShowMoreGrid.jsx';
import CreateModal from '../../../components/common/CreateModal.jsx';
/**
 * ProjectsTab 컴포넌트
 *
 * 프로젝트 목록 불러오기
 * 카드 형태로 표시, 새 프로젝트 생성, 편집, 삭제 기능 제공
 *
 * 주요 기능:
 * - 프로젝트 목록 조회 (API 호출)
 * - 프로젝트 생성, 수정, 삭제
 * - 모달 열기/닫기 및 프로젝트 이름 입력
 * - 로딩/에러/빈 상태 처리
 * - Show More 카드 UI 처리
 */

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

    if (loading || mockState?.loading) return <Loading fullHeight={true} />;
    if (error || mockState?.error) return <ErrorMessage message={error || 'Mock error!'} fullHeight={true} />;
    if (projects.length === 0 || mockState?.empty) return <EmptyState message="No projects found." fullHeight={true} />;

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

    const ProjectCard = ({ project }) => (
        <Card onClick={() => handleProjectClick(project.id)} className={styles.projectCard}>
            <div className={styles.cardContent}>
                <StatusChip status={project.status} className={styles.statusChip} />

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
            <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                {allProjectCards}
            </ShowMoreGrid>

            <CreateModal
                isOpen={isModalOpen}
                onClose={handleModalClose}
                onSubmit={handleCreateProjectSubmit}
                title="Create New Project"
                submitLabel="Create Project"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectName"
            />
            <CreateModal
                isOpen={editModalOpen}
                onClose={handleEditModalClose}
                onSubmit={handleEditProjectSubmit}
                initialValue={editProject?.name || ''}
                title="Edit Project"
                submitLabel="Save Changes"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectNameEdit"
            />
        </>
    );
};

export default ProjectsTab; 