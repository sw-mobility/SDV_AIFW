import React, { useState } from 'react';
import {
  fetchRawDatasets,
  fetchLabeledDatasets,
  uploadDataset,
  downloadDataset,
  getDatasetById,
  updateDataset
} from '../api/datasets.js';

export default {
  title: 'API/Datasets API Playground',
};

export const Playground = () => {
  const [raw, setRaw] = useState([]);
  const [labeled, setLabeled] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState('Image');
  const [newSize, setNewSize] = useState('1GB');
  const [selectedId, setSelectedId] = useState('');
  const [selectedType, setSelectedType] = useState('raw');
  const [updateName, setUpdateName] = useState('');
  const [detail, setDetail] = useState(null);
  const [result, setResult] = useState(null);

  const loadAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRawDatasets();
      setRaw(res.data);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await uploadDataset({ name: newName, type: newType, size: newSize }, selectedType);
      setResult(res);
      await loadAll();
      setNewName('');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await deleteDataset(Number(selectedId), selectedType);
      setResult(res);
      await loadAll();
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
      const res = await updateDataset(Number(selectedId), { name: updateName }, selectedType);
      setResult(res);
      await loadAll();
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
      const res = await getDatasetById(Number(selectedId), selectedType);
      setDetail(res.data);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await downloadDataset(Number(selectedId), selectedType);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 800 }}>
      <h2>Datasets API Playground</h2>
      <button onClick={loadAll} disabled={loading} style={{ marginBottom: 12 }}>모든 데이터셋 불러오기</button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      <div style={{ margin: '16px 0' }}>
        <select value={selectedType} onChange={e => setSelectedType(e.target.value)}>
          <option value="raw">Raw</option>
          <option value="labeled">Labeled</option>
        </select>
        <input
          value={newName}
          onChange={e => setNewName(e.target.value)}
          placeholder="새 데이터셋 이름"
        />
        <input
          value={newType}
          onChange={e => setNewType(e.target.value)}
          placeholder="타입 (Image/Text/Audio/...)"
        />
        <input
          value={newSize}
          onChange={e => setNewSize(e.target.value)}
          placeholder="크기 (예: 1GB)"
        />
        <button onClick={handleUpload} disabled={loading || !newName}>업로드</button>
      </div>
      <div style={{ margin: '16px 0' }}>
        <input
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
          placeholder="데이터셋 ID"
          style={{ width: 80 }}
        />
        <button onClick={handleDetail} disabled={loading || !selectedId}>상세조회</button>
        <input
          value={updateName}
          onChange={e => setUpdateName(e.target.value)}
          placeholder="새 이름"
        />
        <button onClick={handleUpdate} disabled={loading || !selectedId || !updateName}>이름변경</button>
        <button onClick={handleDelete} disabled={loading || !selectedId}>삭제</button>
        <button onClick={handleDownload} disabled={loading || !selectedId}>다운로드</button>
      </div>
      <h3>Raw 데이터셋</h3>
      <ul>
        {raw.map(d => (
          <li key={d.id}>
            <b>{d.name}</b> (ID: {d.id}, 타입: {d.type}, 크기: {d.size}, 상태: {d.status}, 수정일: {d.lastModified})
            <button onClick={() => setSelectedId(d.id)} style={{ marginLeft: 8 }}>선택</button>
          </li>
        ))}
      </ul>
      <h3>Labeled 데이터셋</h3>
      <ul>
        {labeled.map(d => (
          <li key={d.id}>
            <b>{d.name}</b> (ID: {d.id}, 타입: {d.type}, 크기: {d.size}, 라벨수: {d.labelCount}, 상태: {d.status}, 수정일: {d.lastModified})
            <button onClick={() => { setSelectedId(d.id); setSelectedType('labeled'); }} style={{ marginLeft: 8 }}>선택</button>
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