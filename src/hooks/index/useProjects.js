import { useState, useEffect } from 'react';
import { fetchProjects, createProject, deleteProject, updateProject } from '../../api/projects.js';
import { uid } from '../../api/uid.js';

/**
 * 프로젝트 관련 모든 로직을 관리하는 커스텀 훅
 * 
 * @returns {Object} 프로젝트 관련 상태와 핸들러
 */
export const useProjects = () => {
    // 상태 관리
    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    // 모달 상태
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
    const [editProject, setEditProject] = useState(null);
    const [deleteTarget, setDeleteTarget] = useState(null);

    // 프로젝트 목록 조회
    const fetchProjectsList = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetchProjects({ uid });
            setProjects(res.data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 초기 로드
    useEffect(() => {
        fetchProjectsList();
    }, []);

    // 프로젝트 생성
    const handleCreateProject = async (projectData) => {
        try {
            setLoading(true);
            setError(null);
            
            const result = await createProject({ 
                uid, 
                name: projectData.name, 
                description: projectData.description 
            });
            
            setProjects(prev => [result.data, ...prev]);
            setIsCreateModalOpen(false);
            
            // 프로젝트 페이지로 이동 - 생성된 프로젝트의 이름 사용
            const createdProject = result.data;
            let projectName;
            
            if (typeof createdProject === 'string') {
                projectName = createdProject;
            } else if (createdProject && typeof createdProject === 'object') {
                projectName = createdProject.name;
            } else {
                projectName = projectData.name; // 입력한 이름 사용
            }
            
            // 프로젝트 이름이 유효한지 확인
            if (projectName && projectName !== 'undefined' && projectName !== '[object Object]') {
                window.location.href = `/projects/${encodeURIComponent(projectName)}`;
            } else {
                // 입력한 프로젝트 이름으로 이동
                window.location.href = `/projects/${encodeURIComponent(projectData.name)}`;
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 프로젝트 수정
    const handleEditProject = async (projectData) => {
        try {
            setLoading(true);
            setError(null);
            
            const name = typeof projectData === 'string' ? projectData : projectData.name;
            const description = typeof projectData === 'string' ? (editProject.description || '') : projectData.description;
            
            await updateProject({ 
                id: editProject._id || editProject.id, 
                uid, 
                name, 
                description
            });
            
            await fetchProjectsList();
            setIsEditModalOpen(false);
            setEditProject(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 프로젝트 삭제 확인 모달 열기
    const openDeleteConfirm = (project) => {
        setDeleteTarget(project);
        setIsDeleteConfirmOpen(true);
    };

    // 프로젝트 삭제 실행
    const handleDeleteProject = async (projectId) => {
        try {
            setLoading(true);
            setError(null);
            
            await deleteProject({ id: projectId, uid });
            await fetchProjectsList();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // 삭제 확인 모달에서 삭제 실행
    const confirmDelete = async () => {
        if (deleteTarget) {
            await handleDeleteProject(deleteTarget._id || deleteTarget.id);
            setIsDeleteConfirmOpen(false);
            setDeleteTarget(null);
        }
    };

    // 프로젝트 클릭 (페이지 이동)
    const handleProjectClick = (project) => {
        window.location.href = `/projects/${encodeURIComponent(project.name)}`;
    };

    // 모달 핸들러
    const openCreateModal = () => setIsCreateModalOpen(true);
    const closeCreateModal = () => setIsCreateModalOpen(false);
    
    const openEditModal = (project) => {
        setEditProject(project);
        setIsEditModalOpen(true);
    };
    
    const closeEditModal = () => {
        setIsEditModalOpen(false);
        setEditProject(null);
    };

    return {
        // 상태
        projects,
        loading,
        error,
        isCreateModalOpen,
        isEditModalOpen,
        isDeleteConfirmOpen,
        editProject,
        deleteTarget,
        
        // 핸들러
        handleCreateProject,
        handleEditProject,
        openDeleteConfirm,
        confirmDelete,
        handleProjectClick,
        openCreateModal,
        closeCreateModal,
        openEditModal,
        closeEditModal,
        
        // 유틸리티
        fetchProjectsList
    };
}; 