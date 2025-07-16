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
import { PlayCircle, ChevronDown, ChevronUp } from 'lucide-react';

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
  const [paramErrors, setParamErrors] = useState({ epochs: '', lr: '', batchSize: '' });

  // Training
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);

  // Algorithm
  const algorithmOptions = [
    { value: 'yolov8', label: 'YOLOv8' },
    { value: 'algorithm1', label: 'Algorithm 1' },
    { value: 'algorithm2', label: 'Algorithm 2' },
  ];
  const yolov8ParamGroups = [
    {
      group: 'Data Split',
      params: [
        { key: 'train_ratio', label: 'Train Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.7 },
        { key: 'val_ratio', label: 'Val Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.2 },
        { key: 'test_ratio', label: 'Test Ratio', type: 'number', min: 0, max: 1, step: 0.01, default: 0.1 },
      ],
    },
    {
      group: 'Training',
      params: [
        { key: 'epochs', label: 'Epochs', type: 'number', min: 1, max: 1000, step: 1, default: 20 },
        { key: 'batch_size', label: 'Batch Size', type: 'number', min: 1, max: 1024, step: 1, default: 32 },
        { key: 'imgsz', label: 'Image Size', type: 'number', min: 32, max: 4096, step: 32, default: 640 },
        { key: 'device', label: 'Device', type: 'text', default: 'cpu' },
        { key: 'patience', label: 'Patience', type: 'number', min: 0, max: 100, step: 1, default: 50 },
        { key: 'augment', label: 'Augment', type: 'checkbox', default: true },
        { key: 'pretrained', label: 'Pretrained', type: 'checkbox', default: true },
      ],
    },
    {
      group: 'Optimization',
      params: [
        { key: 'optimizer', label: 'Optimizer', type: 'select', options: ['SGD', 'Adam', 'AdamW'], default: 'SGD' },
        { key: 'lr0', label: 'Initial LR', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'lrf', label: 'LR Final', type: 'number', min: 0.00001, max: 1, step: 0.00001, default: 0.01 },
        { key: 'momentum', label: 'Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.937 },
        { key: 'weight_decay', label: 'Weight Decay', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.0005 },
        { key: 'warmup_epochs', label: 'Warmup Epochs', type: 'number', min: 0, max: 100, step: 1, default: 3 },
        { key: 'warmup_momentum', label: 'Warmup Momentum', type: 'number', min: 0, max: 1, step: 0.01, default: 0.8 },
        { key: 'warmup_bias_lr', label: 'Warmup Bias LR', type: 'number', min: 0, max: 1, step: 0.0001, default: 0.1 },
      ],
    },
    {
      group: 'Model/Experiment Management',
      params: [
        { key: 'model', label: 'Model', type: 'text', default: 'yolov8n.pt' },
        { key: 'project', label: 'Project', type: 'text', default: 'runs/train' },
        { key: 'name', label: 'Name', type: 'text', default: '' },
      ],
    },
  ];
  const algorithm1Params = [
    { key: 'paramA', label: 'Param A', type: 'number', min: 0, max: 100, step: 1, default: 10 },
    { key: 'paramB', label: 'Param B', type: 'text', default: '' },
  ];
  const algorithm2Params = [
    { key: 'alpha', label: 'Alpha', type: 'number', min: 0, max: 1, step: 0.01, default: 0.5 },
  ];

  // Algorithm state
  const [algorithm, setAlgorithm] = useState('yolov8');
  const [algoParams, setAlgoParams] = useState({});

  // Accordion open state
  const [openParamGroup, setOpenParamGroup] = useState(0);

  // Code Editor toggle state
  const [showCodeEditor, setShowCodeEditor] = useState(false);

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

  const validateParam = (key, value) => {
    let error = '';
    if (key === 'epochs') {
      if (value < 1 || value > 1000) error = 'Epochs must be between 1 and 1000.';
    } else if (key === 'lr') {
      if (value <= 0 || value > 1) error = 'Learning Rate must be greater than 0 and less than or equal to 1.';
    } else if (key === 'batchSize') {
      if (value < 1 || value > 1024) error = 'Batch Size must be between 1 and 1024.';
    }
    setParamErrors(prev => ({ ...prev, [key]: error }));
    return error === '';
  };

  const handleParamChange = (key, value) => {
    setParams(p => ({ ...p, [key]: value }));
    validateParam(key, value);
  };

  // 기존 handleParamChange와 validateParam는 no-code용이므로, 알고리즘 파라미터용 핸들러 추가
  const handleAlgoParamChange = (key, value) => {
    setAlgoParams(p => ({ ...p, [key]: value }));
  };

  return (
    <div className={styles.container}>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#222", marginBottom: "10px"}}>Training</div>
      {/* Standard/Continual Training Tab Navigation */}
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
      {/* Algorithm Selector (always shown) */}
      <div className={styles.sectionCard}>
        <div className={styles.selectorGroup}>
          <div className={styles.selectorBox}>
            <label className={styles.paramLabel} style={{marginBottom: 4}}>Algorithm</label>
            <select
              className={styles.select}
              value={algorithm}
              onChange={e => {
                setAlgorithm(e.target.value);
                setAlgoParams({});
                setOpenParamGroup(0);
              }}
            >
              {algorithmOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
      {/* Standard/Continual UI */}
      {trainingType === 'standard' ? (
        <>
          {/* Dataset & Snapshot (Standard) */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Dataset</label>
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
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Snapshot</label>
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
          </div>
          {/* Parameters - Accordion UI (Standard) */}
          <div className={styles.paramCardWrap}>
            {algorithm === 'yolov8' ? (
              yolov8ParamGroups.map((group, idx) => (
                <div key={group.group} className={styles.accordionCard}>
                  <div
                    className={styles.accordionHeader + ' ' + (openParamGroup === idx ? styles.accordionOpen : '')}
                    onClick={() => setOpenParamGroup(openParamGroup === idx ? -1 : idx)}
                    tabIndex={0}
                    role="button"
                    aria-expanded={openParamGroup === idx}
                  >
                    <span>{group.group}</span>
                    <span className={styles.accordionArrow}>{openParamGroup === idx ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                  </div>
                  {openParamGroup === idx && (
                    <div className={styles.accordionContent}>
                      {group.params.map(param => (
                        <div className={styles.paramRow} key={param.key}>
                          <label className={styles.paramLabel}>{param.label}</label>
                          {param.type === 'select' ? (
                            <select
                              className={styles.paramInput}
                              value={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, e.target.value)}
                              disabled={isTraining}
                            >
                              {param.options.map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                          ) : param.type === 'checkbox' ? (
                            <input
                              type="checkbox"
                              className={styles.paramInput}
                              checked={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, e.target.checked)}
                              disabled={isTraining}
                            />
                          ) : (
                            <input
                              type={param.type}
                              className={styles.paramInput}
                              value={algoParams[param.key] ?? param.default}
                              min={param.min}
                              max={param.max}
                              step={param.step}
                              placeholder={param.label}
                              disabled={isTraining}
                              onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                            />
                          )}
                          {param.desc && <span className={styles.paramDesc}>{param.desc}</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : algorithm === 'algorithm1' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 1</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm1Params.map(param => (
                      <div className={styles.paramRow} key={param.key}>
                        <label className={styles.paramLabel}>{param.label}</label>
                        <input
                          type={param.type}
                          className={styles.paramInput}
                          value={algoParams[param.key] ?? param.default}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          placeholder={param.label}
                          disabled={isTraining}
                          onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : algorithm === 'algorithm2' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 2</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm2Params.map(param => (
                      <div className={styles.paramRow} key={param.key}>
                        <label className={styles.paramLabel}>{param.label}</label>
                        <input
                          type={param.type}
                          className={styles.paramInput}
                          value={algoParams[param.key] ?? param.default}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          placeholder={param.label}
                          disabled={isTraining}
                          onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </>
      ) : (
        <>
          {/* Continual Learning Info */}
          <div className={styles.sectionCard} style={{ background: '#f8f9fb', border: '1.5px solid #4f8cff', color: '#1a3a5d', marginBottom: 18 }}>
            <b>Continual Learning</b> allows you to update a model incrementally with new data, starting from a previous snapshot. Select a base snapshot and a new dataset to continue training.
          </div>
          {/* Base Snapshot (required) and New Dataset */}
          <div className={styles.sectionCard}>
            <div className={styles.selectorGroup}>
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>Base Snapshot <span style={{color:'#e74c3c'}}>*</span></label>
                <div className={styles.snapshotRow}>
                  <select
                    className={styles.select}
                    value={selectedSnapshot ? selectedSnapshot.id : ''}
                    onChange={e => {
                      const snap = snapshots.find(s => s.id === e.target.value);
                      setSelectedSnapshot(snap);
                    }}
                  >
                    <option value="">Select base snapshot</option>
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
              <div className={styles.selectorBox}>
                <label className={styles.paramLabel} style={{marginBottom: 4}}>New Dataset <span style={{color:'#e74c3c'}}>*</span></label>
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
                    <option value="">Select new dataset</option>
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
            </div>
          </div>
          {/* Parameters - Only show 'Training' group for Continual */}
          <div className={styles.paramCardWrap}>
            {algorithm === 'yolov8' ? (
              yolov8ParamGroups.filter(g => g.group === 'Training').map((group, idx) => (
                <div key={group.group} className={styles.accordionCard}>
                  <div
                    className={styles.accordionHeader + ' ' + (openParamGroup === idx ? styles.accordionOpen : '')}
                    onClick={() => setOpenParamGroup(openParamGroup === idx ? -1 : idx)}
                    tabIndex={0}
                    role="button"
                    aria-expanded={openParamGroup === idx}
                  >
                    <span>{group.group}</span>
                    <span className={styles.accordionArrow}>{openParamGroup === idx ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                  </div>
                  {openParamGroup === idx && (
                    <div className={styles.accordionContent}>
                      {group.params.map(param => (
                        <div className={styles.paramRow} key={param.key}>
                          <label className={styles.paramLabel}>{param.label}</label>
                          {param.type === 'select' ? (
                            <select
                              className={styles.paramInput}
                              value={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, e.target.value)}
                              disabled={isTraining}
                            >
                              {param.options.map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                          ) : param.type === 'checkbox' ? (
                            <input
                              type="checkbox"
                              className={styles.paramInput}
                              checked={algoParams[param.key] ?? param.default}
                              onChange={e => handleAlgoParamChange(param.key, e.target.checked)}
                              disabled={isTraining}
                            />
                          ) : (
                            <input
                              type={param.type}
                              className={styles.paramInput}
                              value={algoParams[param.key] ?? param.default}
                              min={param.min}
                              max={param.max}
                              step={param.step}
                              placeholder={param.label}
                              disabled={isTraining}
                              onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                            />
                          )}
                          {param.desc && <span className={styles.paramDesc}>{param.desc}</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : algorithm === 'algorithm1' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 1</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm1Params.map(param => (
                      <div className={styles.paramRow} key={param.key}>
                        <label className={styles.paramLabel}>{param.label}</label>
                        <input
                          type={param.type}
                          className={styles.paramInput}
                          value={algoParams[param.key] ?? param.default}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          placeholder={param.label}
                          disabled={isTraining}
                          onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : algorithm === 'algorithm2' ? (
              <div className={styles.accordionCard}>
                <div
                  className={styles.accordionHeader + ' ' + (openParamGroup === 0 ? styles.accordionOpen : '')}
                  onClick={() => setOpenParamGroup(openParamGroup === 0 ? -1 : 0)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={openParamGroup === 0}
                >
                  <span>Algorithm 2</span>
                  <span className={styles.accordionArrow}>{openParamGroup === 0 ? <ChevronUp size={18} color="#4f8cff" /> : <ChevronDown size={18} color="#4f8cff" />}</span>
                </div>
                {openParamGroup === 0 && (
                  <div className={styles.accordionContent}>
                    {algorithm2Params.map(param => (
                      <div className={styles.paramRow} key={param.key}>
                        <label className={styles.paramLabel}>{param.label}</label>
                        <input
                          type={param.type}
                          className={styles.paramInput}
                          value={algoParams[param.key] ?? param.default}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          placeholder={param.label}
                          disabled={isTraining}
                          onChange={e => handleAlgoParamChange(param.key, param.type === 'number' ? Number(e.target.value) : e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </>
      )}
      {/* Edit Code/Expert Mode toggle button (always shown) */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', margin: '16px 0 0 0' }}>
        <Button
          variant="secondary"
          onClick={() => setShowCodeEditor(v => !v)}
          style={{ minWidth: 140 }}
        >
          {showCodeEditor ? 'Hide Code Editor' : 'Edit Code (Expert Mode)'}
        </Button>
      </div>
      {/* Collapsible Code Editor Section */}
      {showCodeEditor && (
        <div className={styles.sectionCard} style={{ marginTop: 12 }}>
          <div className={styles.editorCard + ' ' + styles.editorWide}>
            <CodeEditor />
          </div>
        </div>
      )}
      {/* Run Section */}
      <div className={styles.sectionCard}>
        <div className={styles.runCard}>
          <div className={styles.runRow}>
            <div className={styles.runErrorWrap}>
              {error && <ErrorMessage message={error} />}
            </div>
            <Button
              variant="primary-gradient"
              size="medium"
              onClick={handleRunTraining}
              disabled={isTraining}
              icon={isTraining ? <span className={styles.spinner}></span> : <PlayCircle size={20} style={{ marginRight: 6 }} />}
              style={{ minWidth: 150, opacity: isTraining ? 0.6 : 1, cursor: isTraining ? 'not-allowed' : 'pointer' }}
            >
              {isTraining ? 'Running...' : 'Run Training'}
            </Button>
          </div>
        </div>
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
