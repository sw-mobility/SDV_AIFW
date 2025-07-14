import React, { useState } from 'react';
import {
  fetchProjects,
  createProject,
  deleteProject,
  updateProject,
  getProjectById
} from '../api/projects.js';

export default {
  title: 'API/Projects API Playground',
};

export const Playground = () => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [updateName, setUpdateName] = useState('');
  const [detail, setDetail] = useState(null);
  const [result, setResult] = useState(null);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchProjects();
      setProjects(res.data);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await createProject({ name: newProjectName });
      setResult(res);
      await loadProjects();
      setNewProjectName('');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const res = await deleteProject(id);
      setResult(res);
      await loadProjects();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await updateProject(Number(selectedId), { name: updateName });
      setResult(res);
      await loadProjects();
      setUpdateName('');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDetail = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getProjectById(Number(selectedId));
      setDetail(res.data);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 700 }}>
      <h2>Projects API Playground</h2>
      <button onClick={loadProjects} disabled={loading} style={{ marginBottom: 12 }}>프로젝트 목록 불러오기</button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      <div style={{ margin: '16px 0' }}>
        <input
          value={newProjectName}
          onChange={e => setNewProjectName(e.target.value)}
          placeholder="새 프로젝트 이름"
        />
        <button onClick={handleCreate} disabled={loading || !newProjectName}>생성</button>
      </div>
      <div style={{ margin: '16px 0' }}>
        <input
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
          placeholder="프로젝트 ID"
          style={{ width: 80 }}
        />
        <button onClick={handleDetail} disabled={loading || !selectedId}>상세조회</button>
        <input
          value={updateName}
          onChange={e => setUpdateName(e.target.value)}
          placeholder="새 이름"
        />
        <button onClick={handleUpdate} disabled={loading || !selectedId || !updateName}>이름변경</button>
        <button onClick={() => handleDelete(Number(selectedId))} disabled={loading || !selectedId}>삭제</button>
      </div>
      <h3>프로젝트 목록</h3>
      <ul>
        {projects.map(p => (
          <li key={p.id}>
            <b>{p.name}</b> (ID: {p.id}, 상태: {p.status}, 수정일: {p.lastModified})
            <button onClick={() => setSelectedId(p.id)} style={{ marginLeft: 8 }}>선택</button>
          </li>
        ))}
      </ul>
      {detail && (
        <div style={{ margin: '16px 0', background: '#f5f5f5', padding: 12 }}>
          <b>상세정보:</b>
          <pre>{JSON.stringify(detail, null, 2)}</pre>
        </div>
      )}
      {result && (
        <div style={{ margin: '16px 0', background: '#f0f8ff', padding: 12 }}>
          <b>API 응답:</b>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}; 