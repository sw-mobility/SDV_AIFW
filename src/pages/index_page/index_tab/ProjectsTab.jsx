import React, { useState, useEffect } from 'react';
import {Plus, FolderOpen, ChevronDown, Trash2, Pencil} from 'lucide-react';
import Card, { CardGrid } from '../../../components/common/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar } from 'lucide-react';
import { fetchProjects, createProject, deleteProject, updateProject } from '../../../api/projects.js';
import { uid } from '../../../api/uid.js';
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

const ProjectsTab = () => {
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
        fetchProjects({ uid })
            .then(res => setProjects(res.data))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <Loading fullHeight={true} />;
    if (error) return <ErrorMessage message={error} fullHeight={true} />;

    const handleCreateProject = () => {
        setIsModalOpen(true);
    };

    const handleModalClose = () => {
        setIsModalOpen(false);
    };

    const handleCreateProjectSubmit = async (projectData) => {
        try {
            setLoading(true);
            setError(null);
            console.log('Creating project with:', { uid, name: projectData.name, description: projectData.description });
            
            const result = await createProject({ uid, name: projectData.name, description: projectData.description });
            console.log('Project created successfully:', result);
            
            setProjects(prev => [result.data, ...prev]);
        setIsModalOpen(false);
            window.location.href = `/projects/${result.data._id || result.data.id}`;
        } catch (err) {
            console.error('Project creation error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleProjectClick = (projectId) => {
        window.location.href = `/projects/${projectId}`;
    };

    // 프로젝트 목록 새로고침
    const reloadProjects = () => {
        setLoading(true);
        setError(null);
        fetchProjects({ uid })
            .then(res => setProjects(res.data))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    };

    //삭제
    const handleDeleteProject = async (projectId) => {
        setLoading(true);
        setError(null);
        try {
            await deleteProject({ id: projectId, uid });
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
    const handleEditProjectSubmit = async (projectData) => {
        setLoading(true);
        setError(null);
        try {
            const name = typeof projectData === 'string' ? projectData : projectData.name;
            const description = typeof projectData === 'string' ? (editProject.description || '') : projectData.description;
            
            await updateProject({ 
                id: editProject._id || editProject.id, 
                uid, 
                name: name, 
                description: description
            });
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
        <Card onClick={() => handleProjectClick(project._id || project.id)} className={styles.projectCard}>
            <div className={styles.cardContent}>
                <StatusChip status={project.status} className={styles.statusChip} />

                <div className={styles.cardIcon}>
                    <FolderOpen size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {project.name}
                </div>

                <div className={styles.cardDescription}>
                    {project.description ? project.description : <span style={{ color: '#bbb' }}>No description</span>}
                </div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {project.created_at ? new Date(project.created_at).toLocaleDateString() : project.lastModified}
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
                        onClick={e => { e.stopPropagation(); handleDeleteProject(project._id || project.id); }}
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
            <ProjectCard key={project._id || project.id} project={project} />
        ))
    ];
    
    const visibleProjectCards = showMore ? allProjectCards : allProjectCards.slice(0, cardsPerPage);

    return (
        <>
            {projects.length === 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '24px' }}>
                    <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                        {allProjectCards}
                    </ShowMoreGrid>
                    <EmptyState message="No projects found." fullHeight={false} />
                </div>
            ) : (
            <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                {allProjectCards}
            </ShowMoreGrid>
            )}

            <CreateModal
                isOpen={isModalOpen}
                onClose={handleModalClose}
                onSubmit={handleCreateProjectSubmit}
                title="Create New Project"
                submitLabel="Create Project"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectName"
                showDescription={true}
            />
            <CreateModal
                isOpen={editModalOpen}
                onClose={handleEditModalClose}
                onSubmit={handleEditProjectSubmit}
                initialValue={editProject?.name || ''}
                initialDescription={editProject?.description || ''}
                title="Edit Project"
                submitLabel="Save Changes"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectNameEdit"
                showDescription={true}
            />
        </>
    );
};

export default ProjectsTab; 