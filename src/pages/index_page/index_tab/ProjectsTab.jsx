import React, { useState } from 'react';
import {Plus, FolderOpen, ChevronDown, Trash2, Pencil} from 'lucide-react';
import Card, { CardGrid } from '../../../components/ui/Card.jsx';
import styles from '../IndexPage.module.css';
import { Calendar } from 'lucide-react';
import StatusChip from '../../../components/ui/StatusChip.jsx';
import Loading from '../../../components/ui/Loading.jsx';
import ErrorMessage from '../../../components/ui/ErrorMessage.jsx';
import EmptyState from '../../../components/ui/EmptyState.jsx';
import ShowMoreGrid from '../../../components/ui/ShowMoreGrid.jsx';
import CreateModal from '../../../components/ui/CreateModal.jsx';
import DeleteConfirmModal from '../../../components/common/DeleteConfirmModal.jsx';
import { useProjects } from '../../../hooks';
import { SkeletonCard } from '../../../components/ui/Skeleton.jsx';
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
    const cardsPerPage = 8;

    const {
        projects,
        loading,
        error,
        isCreateModalOpen,
        isEditModalOpen,
        isDeleteConfirmOpen,
        editProject,
        deleteTarget,
        handleCreateProject,
        handleEditProject,
        openDeleteConfirm,
        confirmDelete,
        handleProjectClick,
        openCreateModal,
        closeCreateModal,
        openEditModal,
        closeEditModal
    } = useProjects();

    if (error) return <ErrorMessage message={error} fullHeight={true} />;

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const CreateProjectCard = () => (
        <Card onClick={openCreateModal} className={styles.createCard}>
            <div className={styles.createCardContent}>
                <Plus size={32} className={styles.createCardIcon} />
                <div className={styles.createCardText}>
                    Create New Project
                </div>
            </div>
        </Card>
    );

    const ProjectCardSkeleton = () => (
        <SkeletonCard />
    );

    const ProjectCard = ({ project }) => (
        <Card onClick={() => handleProjectClick(project)} className={styles.projectCard}>
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
                        onClick={e => { e.stopPropagation(); openEditModal(project); }}
                        style={{ marginRight: 8 }}
                    >
                        <Pencil size={14} />
                    </button>
                    <button 
                        className={styles.actionButton} 
                        title="Delete"
                        onClick={e => { e.stopPropagation(); openDeleteConfirm(project); }}
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const skeletonCards = Array(7).fill(null).map((_, index) => (
        <ProjectCardSkeleton key={`skeleton-${index}`} />
    ));

    const allProjectCards = [
        <CreateProjectCard key="create" />,
        ...projects.map(project => (
            <ProjectCard key={project._id || project.id} project={project} />
        ))
    ];

    return (
        <>
            {loading ? (
                <ShowMoreGrid cardsPerPage={cardsPerPage} showMore={showMore} onToggleShowMore={handleToggleShowMore}>
                    {[<CreateProjectCard key="create" />, ...skeletonCards]}
                </ShowMoreGrid>
            ) : projects.length === 0 ? (
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
                isOpen={isCreateModalOpen}
                onClose={closeCreateModal}
                onSubmit={handleCreateProject}
                title="Create New Project"
                submitLabel="Create Project"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectName"
                showDescription={true}
            />
            <CreateModal
                isOpen={isEditModalOpen}
                onClose={closeEditModal}
                onSubmit={handleEditProject}
                initialValue={editProject?.name || ''}
                initialDescription={editProject?.description || ''}
                title="Edit Project"
                submitLabel="Save Changes"
                label="Project Name"
                placeholder="Enter project name"
                inputName="projectNameEdit"
                showDescription={true}
            />
            <DeleteConfirmModal
                isOpen={isDeleteConfirmOpen}
                onClose={() => setIsDeleteConfirmOpen(false)}
                onConfirm={confirmDelete}
                title="Delete Project"
                message="Are you sure you want to delete this project?"
                itemName={deleteTarget?.name}
            />
        </>
    );
};

export default ProjectsTab; 