"use client"

import React, { useEffect, useState } from 'react';
import Button from '../../components/common/Button.jsx';
import Modal from '../../components/common/Modal.jsx';
import ProgressBar from '../../components/common/ProgressBar.jsx';
import Loading from '../../components/common/Loading.jsx';
import ErrorMessage from '../../components/common/ErrorMessage.jsx';
import CodeEditor from '../../components/common/CodeEditor.jsx';
import { fetchLabeledDatasets } from '../../api/datasets.js';
import pageStyles from '../index_page/IndexPage.module.css';
import styles from './TrainingPage.module.css';

const mockSnapshots = [
  { id: 'snap1', name: 'MyTrainerSnapshot1', description: 'Default trainer snapshot' },
  { id: 'snap2', name: 'MyTrainerSnapshot2', description: 'Experimental snapshot' },
];

export default function TrainingPage() {
  // Tabs & Mode
  const [trainingType, setTrainingType] = useState('standard'); // 'standard' | 'continual'
  const [mode, setMode] = useState('no-code'); // 'no-code' | 'ide'

  // Dataset
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState(null);

  // Snapshot
  const [snapshots, setSnapshots] = useState(mockSnapshots);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [snapshotModalOpen, setSnapshotModalOpen] = useState(false);

  // Hyperparameters
  const [params, setParams] = useState({ epochs: 20, lr: 0.01, batchSize: 32 });

  // Training
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);

  // Fetch labeled datasets on mount
  useEffect(() => {
    setDatasetLoading(true);
    fetchLabeledDatasets()
      .then(res => {
        setDatasets(res.data);
        setDatasetError(null);
      })
      .catch(e => setDatasetError(e.message))
      .finally(() => setDatasetLoading(false));
  }, []);

  // Training simulation (mock)
  useEffect(() => {
    let interval;
    if (isTraining) {
      setStatus('Training');
      setLogs(l => [...l, 'Training started...']);
      setProgress(0);
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setStatus('Completed');
            setLogs(l => [...l, 'Training completed!']);
            setIsTraining(false);
            return 100;
          }
          setLogs(l => [...l, `Progress: ${prev + 10}%`]);
          return prev + 10;
        });
      }, 700);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

  // Handlers
  const handleRunTraining = () => {
    setError(null);
    if (mode === 'no-code' && (!selectedDataset || !selectedSnapshot)) {
      setError('Please select both a dataset and a snapshot.');
      return;
    }
    setIsTraining(true);
    setStatus('Training');
    setLogs([]);
    setProgress(0);
  };

  const handleParamChange = (key, value) => {
    setParams(p => ({ ...p, [key]: value }));
  };

  return (
    <div className={styles.container}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#222", marginBottom: "10px"}}>Training</div>
      <div className={styles.tabNavigation}>
        <button
          className={`${pageStyles.tabButton} ${trainingType === 'standard' ? pageStyles.activeTab : ''}`}
          onClick={() => setTrainingType('standard')}
        >
          Standard Training
        </button>
        <button
          className={`${pageStyles.tabButton} ${trainingType === 'continual' ? pageStyles.activeTab : ''}`}
          onClick={() => setTrainingType('continual')}
        >
          Continual Training
        </button>
      </div>
      <hr className={styles.tabDivider} />
      {/* Mode toggle for No-Code/IDE */}
      <div className={pageStyles.dataTypeToggle}>
        <button
          className={`${pageStyles.dataTypeButton} ${mode === 'no-code' ? pageStyles.activeDataType : ''}`}
          onClick={() => setMode('no-code')}
        >
          No-Code Mode
        </button>
        <button
          className={`${pageStyles.dataTypeButton} ${mode === 'ide' ? pageStyles.activeDataType : ''}`}
          onClick={() => setMode('ide')}
        >
          IDE Mode
        </button>
      </div>
      <div className={styles.bodySection}>
        {mode === 'no-code' ? (
          <>
            <div className={styles.selectorGroup}>
              <div className={styles.selectorBox}>
                {datasetLoading && <Loading />}
                {datasetError && <ErrorMessage message={datasetError} />}
                {!datasetLoading && !datasetError && (
                  <select
                    className={styles.select}
                    value={selectedDataset ? selectedDataset.id : ''}
                    onChange={e => {
                      const ds = datasets.find(d => d.id === Number(e.target.value));
                      setSelectedDataset(ds);
                    }}
                  >
                    <option value="">Select dataset</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.type}, {ds.size})
                      </option>
                    ))}
                  </select>
                )}
                {selectedDataset && (
                  <div className={styles.datasetInfo}>
                    <div><b>Name:</b> {selectedDataset.name}</div>
                    <div><b>Type:</b> {selectedDataset.type}</div>
                    <div><b>Size:</b> {selectedDataset.size}</div>
                    <div><b>Label Count:</b> {selectedDataset.labelCount}</div>
                  </div>
                )}
              </div>
              <div className={styles.selectorBox}>
                <div className={styles.snapshotRow}>
                  <select
                    className={styles.select}
                    value={selectedSnapshot ? selectedSnapshot.id : ''}
                    onChange={e => {
                      const snap = snapshots.find(s => s.id === e.target.value);
                      setSelectedSnapshot(snap);
                    }}
                  >
                    <option value="">Select snapshot</option>
                    {snapshots.map(snap => (
                      <option key={snap.id} value={snap.id}>{snap.name}</option>
                    ))}
                  </select>
                  <Button variant="secondary" onClick={() => setSnapshotModalOpen(true)}>
                    + Register New
                  </Button>
                </div>
                {selectedSnapshot && (
                  <div className={styles.snapshotInfo}>
                    <div><b>Name:</b> {selectedSnapshot.name}</div>
                    <div><b>Description:</b> {selectedSnapshot.description}</div>
                  </div>
                )}
                <Modal isOpen={snapshotModalOpen} onClose={() => setSnapshotModalOpen(false)} title="Register Snapshot">
                  <div style={{ padding: 16 }}>
                    <div>Snapshot registration feature coming soon.</div>
                    <Button onClick={() => setSnapshotModalOpen(false)} style={{ marginTop: 16 }}>Close</Button>
                  </div>
                </Modal>
              </div>
            </div>
            <div className={styles.paramCard}>
              <div className={styles.paramGrid}>
                <label>
                  Epochs
                  <input
                    type="number"
                    className={styles.input}
                    value={params.epochs}
                    min={1}
                    onChange={e => handleParamChange('epochs', Number(e.target.value))}
                  />
                </label>
                <label>
                  Learning Rate
                  <input
                    type="number"
                    className={styles.input}
                    value={params.lr}
                    step="0.0001"
                    min={0}
                    onChange={e => handleParamChange('lr', Number(e.target.value))}
                  />
                </label>
                <label>
                  Batch Size
                  <input
                    type="number"
                    className={styles.input}
                    value={params.batchSize}
                    min={1}
                    onChange={e => handleParamChange('batchSize', Number(e.target.value))}
                  />
                </label>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className={styles.editorCard + ' ' + styles.editorWide}>
              <CodeEditor />
            </div>
          </>
        )}
        <div className={styles.runCard}>
          <Button
            variant="primary"
            size="large"
            onClick={handleRunTraining}
            disabled={isTraining}
          >
            Run Training
          </Button>
          {error && <ErrorMessage message={error} />}
        </div>
        <h3 className={styles.sectionSubheading}>State and Log</h3>
        <div className={styles.statusCard}>
          <div>
            <ProgressBar percentage={progress} />
          </div>
          <div className={styles.logBox}>
            {logs.length === 0 ? (
              <span className={styles.logEmpty}>No logs yet.</span>
            ) : (
              logs.map((log, i) => <div key={i}>{log}</div>)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
